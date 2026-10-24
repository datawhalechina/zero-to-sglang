# Chapter 3 Forward Pass and Generation (Code Walkthrough)

The [concept chapter](./Chapter3_Forward-Pass-and-Generation.md) introduced the path from token IDs to logits and autoregressive generation. This walkthrough follows mini-sglang's Qwen3 implementation, connecting embeddings, attention, the MLP, and sampling to concrete functions.

The reference is pinned to commit [9a91cfa](https://github.com/sgl-project/mini-sglang/tree/9a91cfafe754aa85daee49998176275667eb58f2). Code blocks are source excerpts or explicitly labeled teaching simplifications that should be read alongside the original files. They omit class initialization, weight loading, and GPU setup. First connect a forward pass to a generation step; Chapter 4 then introduces KV caching.

## 1 Learning Objectives

After completing this chapter, you will be able to:

1. Locate Qwen3's embedding, decoder layers, attention, and LM head.
2. Explain Q/K/V, the causal mask, and residual connections through tensor shapes.
3. Explain why generation reads the last position of each request and when it stops.

## 2 Locate the Source

All paths below are relative to mini-sglang's `python/minisgl/`:

| File and entry point | What to look for |
| --- | --- |
| `models/qwen3.py`: `Qwen3Model`, `Qwen3DecoderLayer` | The model forward pass and each layer's computation order |
| `models/utils.py`: `RopeAttn`, `GatedMLP` | QKV projection, attention, and the gated MLP |
| `layers/attention.py`: `AttentionLayer.forward` | Q/K normalization, RoPE, and the attention backend |
| `layers/norm.py`: `RMSNormFused.forward` | Fused normalization and residual addition |
| `layers/embedding.py`: `ParallelLMHead.forward` | Selecting prediction positions and computing vocabulary scores |
| `engine/engine.py`, `engine/sample.py` | Running the forward pass and choosing the next token |

Read model computation before sampling. Chapter 6 covers how the scheduler organizes multiple requests.

## 3 Follow One Forward Pass

### 3.1 Enter the model with token IDs

`Qwen3Model.forward` looks up embeddings, passes them through the decoder layers, and applies the final normalization:

```python
# models/qwen3.py: Qwen3Model.forward
x = self.embed_tokens.forward(input_ids)
residual: torch.Tensor | None = None
for layer in self.layers.op_list:
    x, residual = layer.forward(x, residual)
return self.norm.forward(x, residual)[0]
```

Here, `input_ids` contains integer indices. The embedding replaces each index with a vector of length `hidden_size`. mini-sglang packs the tokens computed in the current step into a one-dimensional input, giving hidden states of shape `[N,C]`, where `N` is the total token count and `C` is the hidden dimension. The concept chapter's `[B,T,C]` is a regular batch layout used for explanation; keep the two representations distinct.

### 3.2 Computation inside a decoder layer

The body of `Qwen3DecoderLayer.forward` has five lines:

```python
# models/qwen3.py: Qwen3DecoderLayer.forward
x, residual = self.input_layernorm.forward(x, residual)
x = self.self_attn.forward(x)
x, residual = self.post_attention_layernorm.forward(x, residual)
x = self.mlp.forward(x)
return x, residual
```

Why is there no explicit `x + attention(x)`? Addition happens inside `RMSNormFused`. On its first call, it saves the original `x` as the residual and returns the normalized value. Later calls add the current result to the residual and normalize it. The MLP residual addition is deferred to the next layer's input normalization, or to the model's final `self.norm` after the last layer.

Expanded mathematically, the order is still `h = x + Attention(RMSNorm(x))`, followed by `out = h + MLP(RMSNorm(h))`. Passing both `x` and `residual` allows the implementation to fuse these operations.

### 3.3 QKV, RoPE, and causal attention

In `models/utils.py`, Qwen3 uses `RopeAttn` with `has_qk_norm=True`:

```python
# models/utils.py: RopeAttn.forward (memory-release statements omitted)
qkv = self.qkv_proj.forward(x)
o = self.attn.forward(qkv)
return self.o_proj.forward(o)
```

`qkv_proj` combines three linear projections. `AttentionLayer.forward` splits the result into Q, K, and V, applies RMSNorm to each Q/K head's last dimension, and applies RoPE using token positions. V does not receive RoPE. The attention backend computes the result, and the output projection maps the Q heads back to dimension `C`.

The same attention mathematics can be expanded in a regular layout as follows. **This is a teaching simplification, not the GPU backend's source**: assume Q/K normalization and RoPE have already been applied, with `q` shaped `[B,Hq,T,D]`, `k` and `v` shaped `[B,Hkv,T,D]`, and no padding or cache.

```python
import torch

groups = q.shape[1] // k.shape[1]
k = k.repeat_interleave(groups, dim=1)
v = v.repeat_interleave(groups, dim=1)
scores = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
T = q.shape[-2]
allowed = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
scores = scores.masked_fill(~allowed, float("-inf"))
weights = scores.float().softmax(dim=-1).to(q.dtype)
attention = weights @ v
```

GQA requires `Hq` to be divisible by `Hkv`, allowing several Q heads to share K/V. Explicit repetition here makes the matrix operations easier to read. The lower-triangular mask lets position `i` access only positions `0..i`, never future tokens. `head_dim` is configured separately, so do not assume that `C = Hq * D` always holds.

### 3.4 The gated MLP

The core of `GatedMLP.forward` is:

```python
# models/utils.py: GatedMLP.forward (memory-release statements omitted)
gate_up = self.gate_up_proj.forward(x)
y = self.act_fn(gate_up)
return self.down_proj.forward(y)
```

For a Qwen3 configuration using SiLU, the merged projection produces gate and up components. `act_fn` computes `silu(gate) * up`, and the down projection returns to the hidden dimension. The MLP transforms each token's features; attention exchanges information between tokens.

## 4 From Logits to a Generation Loop

### 4.1 Keep the last position of each request

`Qwen3ForCausalLM.forward` passes model output to the LM head:

```python
# models/qwen3.py: Qwen3ForCausalLM.forward
output = self.model.forward(get_global_ctx().batch.input_ids)
logits = self.lm_head.forward(output)
return logits
```

Next, open `ParallelLMHead.forward`. During prefill, it uses `get_last_indices` to select each request's final hidden state before projecting to the vocabulary. Sampling therefore receives one logits row per request, rather than all `[B,T,V]` rows from the conceptual example.

For input `[1,2,3]`, the logits at position 2 predict the token after 3. Earlier positions describe earlier predictions and cannot continue the current prefix.

### 4.2 Put one step into a loop

The following is a **single-request teaching simplification without KV caching**, connecting the concept chapter to the engine. `forward_all(tokens)` represents an initialized model: it takes a nonempty `[1,T]` integer tensor and returns `[1,T,V]` logits for all positions. It is not a mini-sglang API. The model should already be in evaluation mode, and `max_new_tokens` must be a nonnegative integer.

```python
import torch

@torch.inference_mode()
def generate(forward_all, tokens, max_new_tokens, eos_id=None):
    output = []
    for _ in range(max_new_tokens):
        logits = forward_all(tokens)[0, -1]
        token = int(logits.argmax())
        output.append(token)
        tokens = torch.cat((tokens, tokens.new_tensor([[token]])), dim=1)
        if token == eos_id:
            break
    return output
```

Each iteration runs a forward pass, chooses a token, and appends it to the prefix. The function returns newly generated tokens, including EOS; a zero output limit makes no model calls. Recomputing the full prefix exposes the repeated work that KV caching removes in the next chapter.

### 4.3 Find sampling in the engine

The real implementation distributes the loop across the scheduler and engine. After computing logits, `Engine.forward_batch` calls `Sampler.sample`. This excerpt retains its two selection paths:

```python
# engine/sample.py: Sampler.sample (profiling context omitted)
if args.temperatures is None:
    return torch.argmax(logits, dim=-1)
return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)
```

An entirely greedy batch takes argmax directly; other batches use the temperature and Top-k/Top-p sampling path. Sampling chooses tokens, while the scheduler handles length limits and EOS. `eval()` controls layer behavior during inference; `inference_mode()` disables gradient-related recording. They serve different purposes.

## 5 Check Shapes and Causality

Work through a regular-layout example on paper: `B=1,T=3,C=32,Hq=4,Hkv=2,D=8,V=64`. This table describes the teaching snippet in Section 3.3, rather than the engine's packed layout.

| Stage | Tensor shape |
| --- | --- |
| Input / embedding | `[1,3]` / `[1,3,32]` |
| Q / original K and V | `[1,4,3,8]` / `[1,2,3,8]` |
| Attention scores | `[1,4,3,3]` |
| All-position logits / next-token scores | `[1,3,64]` / `[1,64]` |

Check three relationships: changing future input must not change earlier outputs; each generation step appends one token; generation stops at the length limit or a confirmed EOS. Then identify which all-position computations the LM head avoids by selecting the final positions.

## 6 Summary and Exercises

### 6.1 Summary

A forward pass computes prediction scores through embeddings, decoder layers, and the LM head. Generation selects a token from the final position and updates the prefix. mini-sglang organizes these computations with packed input, fused residual operations, and final-position selection. Connect each mathematical step to its tensor layout as you read.

### 6.2 Exercises

1. Why does mini-sglang's prefill LM head not need a vocabulary projection at every input position?
2. Where does residual addition happen if it is absent from `Qwen3DecoderLayer.forward`?
3. Which shapes in Section 5 change if `Hkv` decreases from 2 to 1?
4. Without the causal mask, which output positions could be affected by changes to future tokens?
5. What does generation without caching recompute, and which results could be saved?

## References

- [mini-sglang: Qwen3 model](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/models/qwen3.py)
- [mini-sglang: attention and MLP](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/models/utils.py)
- [mini-sglang: attention layer](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/layers/attention.py)
- [mini-sglang: fused normalization](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/layers/norm.py)
- [mini-sglang: embedding and LM head](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/layers/embedding.py)
- [mini-sglang: engine forward pass](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/engine/engine.py)
- [mini-sglang: sampling](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/engine/sample.py)
- [Chapter 4: KV Cache Optimization](./Chapter4_KV-Cache-Optimization.md)
