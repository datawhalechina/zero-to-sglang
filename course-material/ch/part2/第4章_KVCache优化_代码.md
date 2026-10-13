# 第 4 章 KV Cache 优化（代码走读）

上一章已经完成了一个最小语言模型的前向计算与逐 token 生成。本章把注意力层中最关键的一项推理优化KV Cache实现成一个小型、完整的 Transformer，并用数值测试验证缓存路径与从头计算的路径一致。

本文按 Notebook 形式编写：每个 `python` 都是一个可执行单元。把它们依次复制到同一个 Jupyter Notebook、就可以从上到下运行，即可在 CPU/GPU 上完成本章所有实验。后续第 7、8 章会进一步讨论生产环境中的分页缓存、显存管理与前缀复用。

## 1 本章学习目标

完成本章学习后，你将能够：

1. 找出自回归生成中重复计算发生的位置，并解释 KV Cache 消除了什么计算。
2. 实现多头因果自注意力与 RoPE。
3. 在 prefill 和 decode 两个阶段正确维护每层 K、V 缓存。
4. 用缓存路径与全量重算路径的形状、logits 和生成结果验证实现正确性。
5. 根据模型配置和上下文长度估算 KV Cache 的显存占用。

先安装依赖。

```ipython
%pip install "torch>=2.2,<3"
```

下面的初始化单元固定使用 CPU 与 float32，因此不依赖 GPU。所有后续单元都依赖它定义的名称：

```python
import math
from typing import Optional

import torch
from torch import nn


torch.manual_seed(2026)
DEVICE = torch.device("cpu")
DTYPE = torch.float32

VOCAB_SIZE = 97
HIDDEN_SIZE = 32
NUM_HEADS = 4
NUM_LAYERS = 2
MAX_POSITION_EMBEDDINGS = 64

assert HIDDEN_SIZE % NUM_HEADS == 0
assert (HIDDEN_SIZE // NUM_HEADS) % 2 == 0  # RoPE 按相邻维度两两旋转

print(f"PyTorch: {torch.__version__}")
print(f"device: {DEVICE}, dtype: {DTYPE}")
```

## 2 先观察重复计算

朴素的自回归生成会在每轮把完整序列重新送进模型。假设 prompt 长度为 `prompt_len`，再生成 `new_tokens` 个 token，那么模型输入的 token 数依次增长；KV Cache 之后，prefill 处理完整 prompt，后续每轮 decode 只输入一个新 token。

下面的代码可以计算不使用KV Cache 和使用 KV Cache 计算的次数。

```python
def count_model_input_tokens(prompt_len: int, new_tokens: int) -> tuple[int, int]:
    """返回朴素全量重算与 KV Cache 路径实际送入模型的 token 总数。"""
    if new_tokens < 0:
        raise ValueError("new_tokens must be non-negative")

    full_recompute = sum(prompt_len + step for step in range(new_tokens))
    cached_decode = prompt_len + max(new_tokens - 1, 0)
    return full_recompute, cached_decode


full_tokens, cached_tokens = count_model_input_tokens(prompt_len=8, new_tokens=6)
print(f"不使用kv cache 计算需要计算的token数: {full_tokens} tokens")
print(f"使用Kv Cache 需要计算的token数: {cached_tokens} tokens")
assert full_tokens == 63
assert cached_tokens == 13
```

假设当前已经有 token $x_1, x_2, ..., x_t$，为了预测 $x_{t+1}$，最后一个位置需要计算：

$$
\mathrm{Attention}(Q_t, K_{1:t}, V_{1:t})
$$

朴素循环会再次计算历史 token 的 K、V。它们只由对应 hidden state 与固定投影矩阵决定：

$$
K_i = h_i W_K, \qquad V_i = h_i W_V
$$

KV Cache 在第一次算出历史 K、V 后保留它们；下一轮只计算新 token 的 $K_t, V_t$，再把它们追加到缓存中。

注意：KV Cache **没有消除**新 token 对历史 token 的 attention。新 query 仍需要和全部历史 K 计算分数，并读取历史 V；缓存避免的是历史 token 的重复前向、重复 Q/K/V 投影，以及历史位置之间重复计算的 attention。

## 3 先写一个带缓存的注意力层

为了保证后续单元可以直接复用，本节一次性定义完整的多头因果注意力。缓存中每层 K、V 的逻辑布局固定为 `[batch, num_heads, sequence_length, head_dim]`；RoPE 在 K 写入缓存前应用，因此缓存的 K 已带有其绝对位置信息。

```python
# 单层缓存：K 与 V 的布局均为 [B, H, L, D]。
# B=batch，H=KV head 数，L=当前已缓存的 token 数，D=head_dim。
# L 是唯一会随 Prefill/Decode 增长的维度；缓存属于某一层，不能跨层复用。
KVCache = tuple[torch.Tensor, torch.Tensor]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """将最后一维的相邻元素两两旋转：[-x1, x0, -x3, x2, ...]。"""
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def apply_rope(x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    """对 [B, H, T, D] 的 Q 或 K 应用 RoPE。"""
    if x.ndim != 4:
        raise ValueError(f"expected [B, H, T, D], got {tuple(x.shape)}")
    batch_size, _, seq_len, head_dim = x.shape
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")
    if position_ids.shape != (batch_size, seq_len):
        raise ValueError(
            f"position_ids must have shape {(batch_size, seq_len)}, "
            f"got {tuple(position_ids.shape)}"
        )

    # 每对相邻维度共享一个频率，分别对应二维平面的旋转角度。
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=x.device, dtype=x.dtype) / head_dim))
    angles = position_ids.to(dtype=x.dtype)[..., None] * inv_freq
    # unsqueeze(1) 让每个 attention head 使用同一组位置角度。
    cos = torch.repeat_interleave(angles.cos(), repeats=2, dim=-1).unsqueeze(1)
    sin = torch.repeat_interleave(angles.sin(), repeats=2, dim=-1).unsqueeze(1)
    return x * cos + rotate_half(x) * sin


def make_causal_mask(
    query_len: int,
    key_len: int,
    past_len: int,
    device: torch.device,
) -> torch.Tensor:
    """返回 [query_len, key_len] 的布尔 mask；True 表示该位置不可见。"""
    if query_len <= 0 or key_len <= 0 or past_len < 0:
        raise ValueError("query_len/key_len must be positive and past_len must be non-negative")
    if key_len != past_len + query_len:
        raise ValueError("key_len must equal past_len + query_len")

    # 第 i 个新 query 可见全部历史 key，以及当前 chunk 中不晚于 i 的 key。
    query_positions = torch.arange(query_len, device=device)[:, None]
    key_positions = torch.arange(key_len, device=device)[None, :]
    return key_positions > (past_len + query_positions)


class TinyCausalSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

    def _validate_cache(self, cache: KVCache, batch_size: int) -> int:
        # 缓存来自上一次同一层的 forward；这里只校验结构，不重新计算历史 K/V。
        past_key, past_value = cache
        expected_prefix = (batch_size, self.num_heads)
        # batch/head 不匹配意味着把另一批请求或另一种 attention 配置的 cache 混进来了。
        if past_key.shape[:2] != expected_prefix or past_value.shape[:2] != expected_prefix:
            raise ValueError("cache batch size or number of heads does not match attention")
        # 最后一维不是 head_dim 时，即使能拼接，attention 点积的语义也已错误。
        if past_key.shape[-1] != self.head_dim or past_value.shape[-1] != self.head_dim:
            raise ValueError("cache head_dim does not match attention")
        # K 与 V 必须逐 token 一一对应，长度不同会让 weights @ value 失去对应关系。
        if past_key.shape != past_value.shape:
            raise ValueError("past key and value must have identical shapes")
        # 返回 L；它同时决定本轮 RoPE 的绝对位置与 attention mask 的历史偏移。
        return past_key.size(-2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_value: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[KVCache]]:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [B, T, H]")
        batch_size, query_len, hidden_size = hidden_states.shape
        if hidden_size != self.hidden_size:
            raise ValueError("hidden_states last dimension does not match hidden_size")

        # cache 长度正是本轮新 token 在原序列中的起始绝对位置。
        past_len = 0 if past_key_value is None else self._validate_cache(past_key_value, batch_size)
        expected_positions = torch.arange(
            past_len,
            past_len + query_len,
            device=hidden_states.device,
        ).unsqueeze(0).expand(batch_size, -1)
        if not torch.equal(position_ids, expected_positions):
            raise ValueError("position_ids must start at the cache length and be consecutive")

        # 只有本轮输入会经过 Q/K/V 投影。
        # decode 时 query_len 通常为 1；new_key/new_value 只对应这个新 token。
        query = apply_rope(self._split_heads(self.q_proj(hidden_states)), position_ids)
        new_key = apply_rope(self._split_heads(self.k_proj(hidden_states)), position_ids)
        new_value = self._split_heads(self.v_proj(hidden_states))

        if past_key_value is None:
            # Prefill：没有历史，当前 prompt 计算出的 K/V 直接成为第一份 cache。
            key, value = new_key, new_value
        else:
            past_key, past_value = past_key_value
            # Decode/chunk：past_* 保持原样，只将本轮新增 K/V 接在 L 维末尾。
            # torch.cat 是教学实现；生产系统通常写入预分配 KV Pool，避免复制整个历史。
            key = torch.cat([past_key, new_key], dim=-2)
            value = torch.cat([past_value, new_value], dim=-2)

        # query 只来自本轮输入，key/value 则覆盖“历史 cache + 本轮 token”。
        # 因此新 token 能读取完整上下文，却不用重新投影历史 token。
        scores = (query @ key.transpose(-1, -2)) * (self.head_dim ** -0.5)
        # mask 的 key_len 已包含历史 L 和本轮 query_len；历史 token 永远可见。
        causal_mask = make_causal_mask(query_len, key.size(-2), past_len, hidden_states.device)
        scores = scores.masked_fill(causal_mask[None, None], torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        # 从完整 K/V 中加权读回上下文；返回的 key/value 是下一轮要继续持有的 cache。
        output = self.o_proj(self._merge_heads(weights @ value))
        return output, (key, value) if use_cache else None
```

先对注意力层做一次独立 smoke test。这里没有历史缓存，输入三个 token，因此返回的缓存长度也应为 3：

```python
torch.manual_seed(2026)
attention = TinyCausalSelfAttention(HIDDEN_SIZE, NUM_HEADS).to(device=DEVICE, dtype=DTYPE).eval()
hidden_states = torch.randn(2, 3, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
position_ids = torch.arange(3, device=DEVICE).unsqueeze(0).expand(2, -1)

with torch.inference_mode():
    attention_output, attention_cache = attention(hidden_states, position_ids, use_cache=True)

assert attention_output.shape == (2, 3, HIDDEN_SIZE)
assert attention_cache is not None
assert attention_cache[0].shape == (2, NUM_HEADS, 3, HIDDEN_SIZE // NUM_HEADS)
assert attention_cache[1].shape == attention_cache[0].shape
print("attention smoke test: PASS")
```

## 4 给注意力层增加 KV Cache

### 4.1 缓存里保存什么

上一节返回的 `(key, value)` 就是一层 KV Cache。它们已经按 head 拆分，且 `key` 已应用 RoPE。下面直接检查形状，而不是重新定义另一个注意力类：

```python
cached_key, cached_value = attention_cache
print(f"K cache: {tuple(cached_key.shape)}")
print(f"V cache: {tuple(cached_value.shape)}")
assert cached_key.size(-2) == 3
assert cached_value.size(-2) == 3
```

在本章实现中，K/V 的布局是 `[B, H, L, D]`。其中 `L` 是序列长度，因此追加新 token 时必须沿 `dim=-2` 拼接；沿 batch、head 或 head_dim 拼接都会破坏张量含义。

### 4.2 有历史缓存时的因果掩码

decode 通常一次只输入一个 token，此时单个 query 不会看到未来 query token。但实现还应正确处理“已有历史缓存、一次又输入多个新 token”的 chunk。下面检查 `past_len=4`、`query_len=2` 的 mask：

```python
mask = make_causal_mask(query_len=2, key_len=6, past_len=4, device=DEVICE)
print(mask.to(dtype=torch.int64))

expected_mask = torch.tensor(
    [[False, False, False, False, False, True],
     [False, False, False, False, False, False]],
    device=DEVICE,
)
assert torch.equal(mask, expected_mask)
print("offset causal mask: PASS")
```

第一行对应第一个新 token：它可见四个历史 key 与自己，但不能看第二个新 token。第二行是最后一个新 token，可以看见所有六个 key。这个偏移量正是 `past_len`；如果错误地从 0 开始构造 mask，会把已经存在的历史位置当成未来位置。

### 4.3 真正的多头缓存形状

真实模型的布局可能不同，但要先确认序列维在哪。对本章的 `[B, H, L, D]` 布局，追加操作就是：

```python
new_key = torch.randn(2, NUM_HEADS, 1, HIDDEN_SIZE // NUM_HEADS, device=DEVICE)
appended_key = torch.cat([cached_key, new_key], dim=-2)
assert appended_key.shape == (2, NUM_HEADS, 4, HIDDEN_SIZE // NUM_HEADS)
print("append along sequence dimension: PASS")
```

生产模型中，query head 数与 KV head 数可能不同；显存估算和缓存形状应使用 KV head 数，而不是 query head 数。本章小模型使用普通多头注意力，因此两者相等。

## 5 模型级缓存：每层各存一份

Transformer 的每一层都有独立 K/V 投影，不能共享同一份缓存。下面定义一个最小但完整的模型：embedding、预归一化、残差、attention、MLP 和 LM head 都包含在内；`TinyTransformer.forward` 会为每一层传回对应 cache，并从 cache 长度推导 RoPE 的绝对位置。

```python
class TinyTransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: int = 2):
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn = TinyCausalSelfAttention(hidden_size, num_heads)
        self.mlp_norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * mlp_ratio),
            nn.GELU(),
            nn.Linear(hidden_size * mlp_ratio, hidden_size),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_value: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[KVCache]]:
        attention_output, present_key_value = self.attn(
            self.attn_norm(hidden_states),
            position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + attention_output
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        return hidden_states, present_key_value


class TinyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [TinyTransformerBlock(hidden_size, num_heads) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def _past_len(self, past_key_values: Optional[list[KVCache]], batch_size: int) -> int:
        if past_key_values is None:
            return 0
        if len(past_key_values) != self.num_layers:
            raise ValueError("past_key_values must contain one cache per layer")

        lengths = []
        for layer, cache in zip(self.layers, past_key_values):
            # 第 layer_idx 层只校验/读取第 layer_idx 份 cache，避免层间 K/V 串用。
            lengths.append(layer.attn._validate_cache(cache, batch_size))
        # 所有层必须停在同一个 token 位置，才能共用同一组 position_ids。
        if len(set(lengths)) != 1:
            raise ValueError("all layer caches must have the same sequence length")
        return lengths[0]

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[list[KVCache]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[list[KVCache]]]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, T]")
        batch_size, query_len = input_ids.shape
        if query_len == 0:
            raise ValueError("input_ids must contain at least one token")

        # 全部层共享同一批绝对位置，但各层只能读取自己的 K/V cache。
        past_len = self._past_len(past_key_values, batch_size)
        position_ids = torch.arange(
            past_len,
            past_len + query_len,
            device=input_ids.device,
        ).unsqueeze(0).expand(batch_size, -1)

        hidden_states = self.embed_tokens(input_ids)
        next_key_values = []
        for layer_idx, layer in enumerate(self.layers):
            # 取同一索引的 cache：第 0 层绝不能使用第 1 层产生的 K/V。
            past = None if past_key_values is None else past_key_values[layer_idx]
            hidden_states, present = layer(
                hidden_states,
                position_ids,
                past_key_value=past,
                use_cache=use_cache,
            )
            if use_cache:
                assert present is not None
                # 按层序保留 present，下一轮会原样传回对应 layer_idx。
                next_key_values.append(present)

        logits = self.lm_head(self.final_norm(hidden_states))
        return logits, next_key_values if use_cache else None
```

创建固定模型与固定 prompt。后面所有实验都复用这两个对象；模型处于 `eval()`，因此没有 dropout 带来的随机性：

```python
torch.manual_seed(2026)
model = TinyTransformer(
    vocab_size=VOCAB_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
).to(device=DEVICE, dtype=DTYPE).eval()

prompt_ids = torch.tensor([[5, 11, 23, 47]], device=DEVICE, dtype=torch.long)
print(f"prompt_ids: {prompt_ids.tolist()}")
```

先验证模型级缓存：每层恰好一份 K/V，且 prefill 后所有层的缓存长度都等于 prompt 长度：

```python
with torch.inference_mode():
    prompt_logits, prompt_cache = model(prompt_ids, use_cache=True)

assert prompt_cache is not None
assert len(prompt_cache) == NUM_LAYERS
for layer_key, layer_value in prompt_cache:
    assert layer_key.shape == (1, NUM_HEADS, prompt_ids.size(1), HIDDEN_SIZE // NUM_HEADS)
    assert layer_value.shape == layer_key.shape
print("model-level cache invariant: PASS")
```

## 6 Prefill 与 Decode：两种调用方式

### 6.1 Prefill：处理整个 prompt

Prefill 一次处理完整 prompt，既建立每层缓存，也产生 prompt 最后一个位置的 logits，用于选择第一个输出 token：

```python
with torch.inference_mode():
    prefill_logits, prefill_cache = model(prompt_ids, use_cache=True)

next_token = prefill_logits[:, -1].argmax(dim=-1, keepdim=True)
assert prefill_cache is not None
assert all(cache[0].size(-2) == prompt_ids.size(1) for cache in prefill_cache)
print(f"first generated token: {next_token.tolist()}")
print(f"prefill cache length: {prefill_cache[0][0].size(-2)}")
```

### 6.2 Decode：之后每次只处理一个 token

现在只把刚选出的 `next_token` 送给模型，而不是把完整 `prompt_ids + next_token` 重送一遍。模型从 `prefill_cache` 长度推导其绝对位置，并把新的 K/V 追加到每层缓存：

```python
with torch.inference_mode():
    decode_logits, decode_cache = model(
        next_token,
        past_key_values=prefill_cache,
        use_cache=True,
    )

assert decode_cache is not None
assert all(cache[0].size(-2) == prompt_ids.size(1) + 1 for cache in decode_cache)
assert decode_logits.shape == (1, 1, VOCAB_SIZE)
print(f"decode cache length: {decode_cache[0][0].size(-2)}")
```

下面给出完整生成函数。`generate_with_cache` 只在 prefill 时输入完整 prompt，之后每轮只输入上一个新 token；`generate_full_recompute` 是正确性基线，每轮都从头计算完整序列。二者使用相同的 greedy 采样，因此最终 token ids 应完全一致：

```python
@torch.inference_mode()
def generate_with_cache(
    model: TinyTransformer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    generated = input_ids.clone()
    if max_new_tokens == 0:
        return generated

    logits, cache = model(generated, use_cache=True)  # Prefill：整段 prompt 只计算一次
    next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
    generated = torch.cat([generated, next_token], dim=1)

    for _ in range(max_new_tokens - 1):
        if eos_token_id is not None and bool((next_token == eos_token_id).all()):
            break
        # Decode：cache 已包含历史，因此本轮只输入上一步生成的 token。
        logits, cache = model(next_token, past_key_values=cache, use_cache=True)
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

    return generated


@torch.inference_mode()
def generate_full_recompute(
    model: TinyTransformer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        logits, _ = model(generated, use_cache=False)
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        if eos_token_id is not None and bool((next_token == eos_token_id).all()):
            break

    return generated


cached_generation = generate_with_cache(model, prompt_ids, max_new_tokens=5)
full_generation = generate_full_recompute(model, prompt_ids, max_new_tokens=5)
assert torch.equal(cached_generation, full_generation)
print(f"generated ids: {cached_generation.tolist()}")
print("cached and full greedy generation: PASS")
```

如果把完整 `generated` 传给已经带有 `cache` 的模型，历史 token 会既存在于 cache 中又再次作为新输入，导致 cache 长度、绝对位置和 attention 可见范围都错误。Decode 的输入应始终是**刚刚生成的一个 token**。

## 7 Position ids 与 RoPE：缓存后不能从 0 重新编号

RoPE 作用于 Q 与 K。历史 K 被缓存时已经按历史绝对位置旋转；因此 decode 时的新 Q/K 必须从 `past_len` 开始编号。本章模型在 `TinyTransformer.forward` 内部完成此事，下面用同样的公式观察位置：

```python
past_len = prompt_ids.size(1)
query_len = 1
positions = torch.arange(past_len, past_len + query_len, device=DEVICE).unsqueeze(0)
assert torch.equal(positions, torch.tensor([[4]], device=DEVICE))
print(f"decode position_ids: {positions.tolist()}")
```

用固定新 token 比较“缓存 decode”与“完整序列从头计算”的最后一个 logits。两条路径的矩阵运算分组可能略有不同，因此比较浮点 logits 时使用 `assert_close`，不使用逐元素相等：

```python
fixed_next_token = torch.tensor([[42]], device=DEVICE, dtype=torch.long)

with torch.inference_mode():
    _, cache_for_position_test = model(prompt_ids, use_cache=True)
    cached_logits, _ = model(
        fixed_next_token,
        past_key_values=cache_for_position_test,
        use_cache=True,
    )
    full_logits, _ = model(torch.cat([prompt_ids, fixed_next_token], dim=1), use_cache=False)

torch.testing.assert_close(cached_logits[:, -1], full_logits[:, -1], rtol=1e-5, atol=1e-6)
print("RoPE absolute-position check: PASS")
```

若每轮 decode 从位置 0 重新编号，新 token 的 Q/K 会被旋转到错误位置；即使张量形状仍然正确，也不能与完整计算路径对齐。

## 8 验证：缓存结果应与全量重算一致

### 8.1 单 token、一轮一轮与 chunk 三种测试

先把三类验证封装成可执行函数。它们共同确认：缓存长度随输入增长；每层仍拥有同长度的 K/V；cache 路径产生的 logits 与完整序列从头计算的 logits 一致：

```python
@torch.inference_mode()
def assert_cache_matches_full(
    model: TinyTransformer,
    prefix_ids: torch.Tensor,
    suffix_ids: torch.Tensor,
) -> None:
    """逐 token 追加 suffix，并在每一步比较 cached 与 full logits。"""
    _, cache = model(prefix_ids, use_cache=True)
    full_ids = prefix_ids.clone()

    for step in range(suffix_ids.size(1)):
        token = suffix_ids[:, step : step + 1]
        cached_logits, cache = model(token, past_key_values=cache, use_cache=True)
        full_ids = torch.cat([full_ids, token], dim=1)
        full_logits, _ = model(full_ids, use_cache=False)

        torch.testing.assert_close(
            cached_logits[:, -1],
            full_logits[:, -1],
            rtol=1e-5,
            atol=1e-6,
        )
        assert cache is not None
        expected_len = full_ids.size(1)
        for key, value in cache:
            assert key.size(-2) == expected_len
            assert value.shape == key.shape


@torch.inference_mode()
def assert_chunked_cache_matches_full(
    model: TinyTransformer,
    prefix_ids: torch.Tensor,
    chunk_ids: torch.Tensor,
) -> None:
    """已有 cache 时一次输入多个 token，验证 offset causal mask。"""
    _, cache = model(prefix_ids, use_cache=True)
    chunked_logits, chunked_cache = model(chunk_ids, past_key_values=cache, use_cache=True)
    full_logits, _ = model(torch.cat([prefix_ids, chunk_ids], dim=1), use_cache=False)

    torch.testing.assert_close(chunked_logits, full_logits[:, -chunk_ids.size(1) :], rtol=1e-5, atol=1e-6)
    assert chunked_cache is not None
    assert all(item[0].size(-2) == prefix_ids.size(1) + chunk_ids.size(1) for item in chunked_cache)


single_suffix = torch.tensor([[42]], device=DEVICE, dtype=torch.long)
multiple_suffix = torch.tensor([[7, 13, 29]], device=DEVICE, dtype=torch.long)
chunk_suffix = torch.tensor([[31, 37]], device=DEVICE, dtype=torch.long)

assert_cache_matches_full(model, prompt_ids, single_suffix)
assert_cache_matches_full(model, prompt_ids, multiple_suffix)
assert_chunked_cache_matches_full(model, prompt_ids, chunk_suffix)
print("one-token, multi-step, and chunked cache tests: PASS")
```

### 8.2 失败时如何排查

若上面的验证失败，按以下顺序检查：

1. 新 K/V 是否沿**序列维**追加，而不是沿 batch、head 或 head_dim 追加；
2. cache 中每层的 K/V 是否都交回了同一层；
3. prefill 和带历史 cache 的 chunk 是否都使用了正确的 offset causal mask；
4. RoPE 是否在新 K 写入 cache 前应用，且新 token 的位置是否从 `past_len` 开始；
5. decode 是否只传入一个新 token；
6. 模型是否处于 `eval()`，并关闭或避免了 dropout；
7. 是否使用 `assert_close` 比较 logits，而不是对浮点值做逐位相等判断。

## 9 显存估算：为什么上下文和并发受 KV Cache 限制

对有 $N$ 层、batch size 为 $B$、缓存长度为 $L$、KV head 数为 $H_{kv}$、head dim 为 $D$ 的模型，K/V 缓存近似占用：

$$
\mathrm{KV\ bytes} = 2 \times N \times B \times L \times H_{kv} \times D \times s
$$

其中 $2$ 表示 K 和 V，$s$ 是每个元素的字节数。下面函数只做容量估算，不会真的在 CPU 分配这些张量：

```python
def estimate_kv_cache_gib(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int,
    bytes_per_element: int = 2,
) -> float:
    if min(num_layers, num_kv_heads, head_dim, seq_len, batch_size, bytes_per_element) <= 0:
        raise ValueError("all dimensions and bytes_per_element must be positive")
    total_bytes = (
        2
        * num_layers
        * batch_size
        * seq_len
        * num_kv_heads
        * head_dim
        * bytes_per_element
    )
    return total_bytes / 1024**3


estimate = estimate_kv_cache_gib(
    num_layers=32,
    num_kv_heads=32,
    head_dim=128,
    seq_len=4096,
    batch_size=16,
)
assert estimate == 32.0
print(f"estimated KV Cache: {estimate:.1f} GiB")
```

这里的 `num_kv_heads` 不一定等于 query head 数。使用较少 KV heads 的架构会减少缓存显存；估算时必须使用实际 KV head 数。上例只计算 KV Cache，不包括模型权重、激活、attention 临时工作区和运行时预留空间。

## 10 总结与测试题

### 10.1 课程总结

KV Cache 对每一层 attention 保存历史 token 的 K、V。Prefill 一次计算完整 prompt 并建立缓存；之后的 Decode 每轮只计算新 token 的 Q/K/V，让新 Q 读取缓存中的历史 K/V 与本轮新 K/V。

它避免了历史 token 的重复前向与投影，却不消除新 query 对整个历史上下文的 attention 读取。因此 KV Cache 用显存换计算：上下文越长、并发越高，缓存越大；也正因如此，后续还需要分页分配、调度与前缀复用等优化。

### 10.2 测试题

1. 为什么 Decode 阶段只传入一个新 token，仍能注意到完整 prompt 和历史输出？
2. `past_key_values` 为什么必须按层保存，而不能让所有 Transformer 层共享同一份 K/V？
3. 对 `[B, H, L, D]` 布局，新增 token 时应沿哪一维追加 K/V？为什么？
4. 使用 RoPE 时，为什么 decode 新 token 的绝对位置必须从 `past_len` 开始？

## 参考资料

- [Transformer 原始论文](https://arxiv.org/abs/1706.03762)
- [RoPE 论文](https://arxiv.org/abs/2104.09864)
- [mini-sglang 源码](https://github.com/sgl-project/mini-sglang)
