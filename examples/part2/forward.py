"""Chapter 3: dense Qwen3, written with ordinary PyTorch operations.

Architecture reference: Qwen3 / Transformers v4.57.1. Weight names follow
that implementation so tests can load identical weights. No HF forward or
generate is called by this model. Only standard full attention is supported.
"""

from dataclasses import dataclass, fields
import math

import torch
from torch import nn
from torch.nn import functional as F

KVCache = list[tuple[torch.Tensor, torch.Tensor]]  # per layer: [B, Hkv, S, D]


@dataclass
class Config:
    vocab_size: int = 64
    hidden_size: int = 32
    intermediate_size: int = 64
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 8
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 256
    tie_word_embeddings: bool = False

    def __post_init__(self):
        sizes = [self.vocab_size, self.hidden_size, self.intermediate_size,
                 self.num_hidden_layers, self.num_attention_heads,
                 self.num_key_value_heads, self.head_dim, self.max_position_embeddings]
        if any(x <= 0 for x in sizes):
            raise ValueError("model dimensions must be positive")
        if self.head_dim % 2 or self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("RoPE needs even head_dim; Hq must be divisible by Hkv")

    @classmethod
    def from_hf(cls, cfg):
        if cfg.model_type != "qwen3" or cfg.hidden_act != "silu":
            raise ValueError("this example supports dense Qwen3 with SiLU only")
        if cfg.attention_bias or cfg.rope_scaling or cfg.use_sliding_window:
            raise ValueError("bias, scaled RoPE, and sliding attention are not implemented")
        return cls(**{f.name: getattr(cfg, f.name) for f in fields(cls)})


class RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        fp32 = x.float()
        normalized = fp32 / torch.sqrt(fp32.square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight


def rope(x, positions, theta):
    # Qwen uses two halves, not interleaved even/odd coordinates.
    dim = x.shape[-1]
    frequencies = theta ** (-torch.arange(0, dim, 2, device=x.device).float() / dim)
    angles = positions.float()[..., None] * frequencies
    angles = torch.cat((angles, angles), dim=-1)[:, None]
    left, right = x.chunk(2, dim=-1)
    rotated = torch.cat((-right, left), dim=-1)
    return x * angles.cos().to(x.dtype) + rotated * angles.sin().to(x.dtype)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        for name, heads in (("q_proj", c.num_attention_heads),
                            ("k_proj", c.num_key_value_heads),
                            ("v_proj", c.num_key_value_heads)):
            setattr(self, name, nn.Linear(c.hidden_size, heads * c.head_dim, bias=False))
        self.o_proj = nn.Linear(c.num_attention_heads * c.head_dim, c.hidden_size, bias=False)
        self.q_norm = RMSNorm(c.head_dim, c.rms_norm_eps)
        self.k_norm = RMSNorm(c.head_dim, c.rms_norm_eps)

    def forward(self, x, positions, allowed, past):
        b, t, _ = x.shape
        def heads(projection):
            return projection(x).view(b, t, -1, self.c.head_dim).transpose(1, 2)
        q = rope(self.q_norm(heads(self.q_proj)), positions, self.c.rope_theta)
        k = rope(self.k_norm(heads(self.k_proj)), positions, self.c.rope_theta)
        v = heads(self.v_proj)
        if past is not None:
            k, v = torch.cat((past[0], k), dim=2), torch.cat((past[1], v), dim=2)
        cache = (k, v)  # Cache before expanding GQA heads.
        groups = self.c.num_attention_heads // self.c.num_key_value_heads
        k, v = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.c.head_dim)
        scores = scores.masked_fill(~allowed, float("-inf"))
        weights = scores.float().softmax(-1).to(x.dtype)
        attended = (weights @ v).transpose(1, 2).reshape(b, t, -1)
        return self.o_proj(attended), cache


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down_proj = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)

    def forward(self, x):
        gated = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(gated)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.self_attn, self.mlp = Attention(c), MLP(c)
        self.input_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)

    def forward(self, x, positions, allowed, past):
        attention, cache = self.self_attn(self.input_layernorm(x), positions, allowed, past)
        x = x + attention
        return x + self.mlp(self.post_attention_layernorm(x)), cache


class Qwen3(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = c = config or Config()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size)
        self.model.layers = nn.ModuleList(Block(c) for _ in range(c.num_hidden_layers))
        self.model.norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        if c.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, attention_mask=None, cache=None, use_cache=False):
        b, t = input_ids.shape
        if t == 0:
            raise ValueError("input must contain at least one token")
        past_len = 0 if cache is None else cache[0][0].shape[2]
        if cache is not None and len(cache) != len(self.model.layers):
            raise ValueError("cache must contain one entry per layer")
        size = past_len + t
        mask = (torch.ones((b, size), dtype=torch.bool, device=input_ids.device)
                if attention_mask is None else attention_mask.bool())
        if mask.shape != (b, size) or not mask.any(-1).all():
            raise ValueError("attention_mask must cover past + current tokens; no empty rows")
        if mask.sum(-1).max() > self.config.max_position_embeddings:
            raise ValueError("context limit exceeded")
        positions = (mask.long().cumsum(-1) - 1).clamp_min(0)[:, past_len:]
        keys = torch.arange(size, device=input_ids.device)
        queries = torch.arange(past_len, size, device=input_ids.device)
        allowed = (keys[None, :] <= queries[:, None])[None, None] & mask[:, None, None, :]
        # Padded query rows are discarded. Give them one key to avoid NaN softmax.
        allowed = allowed | ((keys[None, :] == queries[:, None])[None, None]
                             & ~mask[:, None, past_len:, None])
        x, updated = self.model.embed_tokens(input_ids), []
        for index, layer in enumerate(self.model.layers):
            x, kv = layer(x, positions, allowed, None if cache is None else cache[index])
            if use_cache:
                updated.append(kv)
        return self.lm_head(self.model.norm(x)), updated if use_cache else None


def probabilities(logits, temperature=1.0, top_k=0, top_p=1.0):
    """Apply temperature, then top-k, then nucleus filtering; preserve cutoff token."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive; greedy uses argmax")
    if not 0 < top_p <= 1 or top_k < 0:
        raise ValueError("require 0 < top_p <= 1 and top_k >= 0")
    scores, order = (logits.float() / temperature).sort(descending=True, dim=-1)
    if top_k:
        scores[..., min(top_k, scores.shape[-1]):] = -torch.inf
    mass = scores.softmax(-1)
    scores = scores.masked_fill(mass.cumsum(-1) - mass >= top_p, -torch.inf)
    return torch.zeros_like(scores).scatter(-1, order, scores.softmax(-1))


@torch.inference_mode()
def generate(model, prompt, max_new_tokens=16, eos_id=None, temperature=0.0,
             top_k=0, top_p=1.0, use_cache=False, generator=None):
    """Single-request reference; returns NEW tokens only, including terminal EOS."""
    if not prompt or max_new_tokens < 0 or not math.isfinite(temperature) or temperature < 0:
        raise ValueError("nonempty prompt, nonnegative length and temperature required")
    if len(prompt) + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("prompt + requested output exceeds context limit")
    device = next(model.parameters()).device
    tokens = torch.tensor([prompt], dtype=torch.long, device=device)
    output, cache = [], None
    for _ in range(max_new_tokens):
        inputs = tokens if cache is None else tokens[:, -1:]
        logits, cache = model(inputs, cache=cache, use_cache=use_cache)
        last = logits[0, -1]
        token = (last.argmax() if temperature == 0 else torch.multinomial(
            probabilities(last, temperature, top_k, top_p), 1, generator=generator)[0])
        output.append(int(token))
        if int(token) == eos_id:
            break
        tokens = torch.cat((tokens, token.reshape(1, 1)), dim=1)
    return output


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="optional Qwen3 model ID or local checkpoint")
    parser.add_argument("--prompt", default="用一句话解释自回归生成。")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--cache", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(7)
    if args.model:
        from .runtime import load_pretrained
        model, tokenizer = load_pretrained(args.model)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False)
        output = generate(model, prompt, args.max_new_tokens, tokenizer.eos_token_id,
                          use_cache=args.cache)
        print(tokenizer.decode(output, skip_special_tokens=True))
    else:
        model = Qwen3().eval()
        output = generate(model, [1, 2, 3], args.max_new_tokens, use_cache=args.cache)
        print("Random tiny model; token IDs demonstrate mechanics, not language ability:")
        print(output)


if __name__ == "__main__":
    main()
