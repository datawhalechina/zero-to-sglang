"""Chapter 10: exact linear speculative sampling and greedy verification.

Full-prefix forward passes intentionally avoid speculative KV rollback. This is
an algorithm reference, not a speed benchmark or an EAGLE implementation.
"""

from dataclasses import dataclass, field
import math

import torch

from .forward import Config, Qwen3, generate, probabilities


@dataclass
class Stats:
    target_calls: int = 0
    draft_calls: int = 0
    proposed: int = 0
    accepted: int = 0
    rounds: list[dict] = field(default_factory=list)


def draw(probs, generator=None):
    return int(torch.multinomial(probs, 1, generator=generator)[0])


def rejection_distribution(p, q):
    residual = (p - q).clamp_min(0)
    mass = residual.sum()
    if not torch.isfinite(mass) or mass <= 0:
        raise ValueError("rejection has no positive residual mass")
    return residual / mass


def accept_token(p, q, token, generator=None):
    # token was sampled from q, hence q[token] must be positive.
    if q[token] <= 0:
        raise ValueError("draft token must have positive proposal probability")
    probability = min(1.0, float(p[token] / q[token]))
    return float(torch.rand((), device=p.device, generator=generator)) < probability


@torch.inference_mode()
def speculative_generate(target, draft, prompt, max_new_tokens=16, draft_steps=4,
                         eos_id=None, temperature=0.0, top_k=0, top_p=1.0,
                         generator=None):
    if not prompt or max_new_tokens < 0 or draft_steps < 1:
        raise ValueError("nonempty prompt, nonnegative output length, positive draft_steps required")
    if not math.isfinite(temperature) or temperature < 0 or not 0 < top_p <= 1 or top_k < 0:
        raise ValueError("invalid sampling parameters")
    if target.config.vocab_size != draft.config.vocab_size:
        raise ValueError("draft and target must share vocabulary size AND token meanings")
    # Equal sizes are necessary, not sufficient: callers must ensure identical tokenizers.
    context_limit = min(target.config.max_position_embeddings, draft.config.max_position_embeddings)
    if len(prompt) + max_new_tokens > context_limit:
        raise ValueError("prompt + requested output exceeds context limit")
    device = next(target.parameters()).device
    if next(draft.parameters()).device != device:
        raise ValueError("this teaching implementation requires both models on one device")
    tokens, output, stats = list(prompt), [], Stats()

    def commit(token):
        tokens.append(token)
        output.append(token)
        return token == eos_id or len(output) == max_new_tokens

    while len(output) < max_new_tokens:
        proposals, proposal_probs = [], []
        remaining = max_new_tokens - len(output)
        for _ in range(min(draft_steps, remaining)):
            ids = torch.tensor([tokens + proposals], device=device)
            logits, _ = draft(ids)
            stats.draft_calls += 1
            q = (None if temperature == 0 else probabilities(logits[0, -1], temperature, top_k, top_p))
            token = int(logits[0, -1].argmax()) if q is None else draw(q, generator)
            proposals.append(token)
            proposal_probs.append(q)
            if token == eos_id:
                break

        # Causal row L-1 predicts proposal[0]; row L+k-1 predicts the bonus token.
        logits, _ = target(torch.tensor([tokens + proposals], device=device))
        stats.target_calls += 1
        stats.proposed += len(proposals)
        verification = logits[0, len(tokens)-1:len(tokens)+len(proposals)]
        accepted, rejected, terminal = 0, False, False
        for i, token in enumerate(proposals):
            if temperature == 0:
                replacement = int(verification[i].argmax())
                accept = token == replacement
            else:
                p = probabilities(verification[i], temperature, top_k, top_p)
                accept = accept_token(p, proposal_probs[i], token, generator)
                replacement = None if accept else draw(rejection_distribution(p, proposal_probs[i]), generator)
            if accept:
                accepted += 1
                stats.accepted += 1
                terminal = commit(token)
            else:
                rejected = True
                terminal = commit(replacement)
            if terminal or rejected:
                break  # Rejecting a token invalidates all subsequent draft conditionals.
        if not terminal and not rejected:
            bonus = (int(verification[-1].argmax()) if temperature == 0 else draw(
                probabilities(verification[-1], temperature, top_k, top_p), generator))
            terminal = commit(bonus)
        stats.rounds.append({"proposals": proposals, "accepted": accepted, "rejected": rejected})
        if terminal:
            break
    return output, stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--draft-steps", type=int, default=4)
    args = parser.parse_args()
    torch.manual_seed(7)
    torch.set_num_threads(1)
    target = Qwen3().eval()
    draft = Qwen3(Config(num_hidden_layers=1)).eval()
    output, stats = speculative_generate(target, draft, [1, 2, 3], 16,
                                         args.draft_steps, temperature=args.temperature)
    print("Random tiny models; generated token IDs:", output)
    if args.temperature == 0:
        baseline = generate(target, [1, 2, 3], 16)
        print("Matches target-only greedy:", output == baseline)
    print(f"target_calls={stats.target_calls} draft_calls={stats.draft_calls} "
          f"accepted/proposed={stats.accepted}/{stats.proposed}")
    print("Call counts are NOT a measured latency or throughput speedup.")


if __name__ == "__main__":
    main()
