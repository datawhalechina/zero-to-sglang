# Chapter 10 Speculative Decoding (Code Walkthrough)

Ordinary generation confirms one new token per target-model call. Speculative decoding proposes several candidates before the target verifies them. Building on the [concept chapter](./Chapter10_Speculative-Decoding.md), this walkthrough follows drafting, verification, and commitment, focusing on position alignment, the first rejection, and stopping conditions.

The mini-sglang commit `9a91cfa` used in earlier chapters has no speculative-decoding module, so source references here use SGLang commit [1291f317](https://github.com/sgl-project/sglang/tree/1291f3175d3b7614821884e9653985fcd3c3dae8). Code blocks are labeled as simplified source or teaching algorithm fragments. Read them with the stated interfaces; they do not form a standalone inference program.

## 1 Learning Objectives

After completing this chapter, you will be able to:

1. Locate source entry points for the draft model, verification inputs, and execution flow.
2. Align each proposal with its prediction row and handle the first rejection and extra token.
3. Explain acceptance probabilities, rejection correction, and the cache state that must be restored.

## 2 Locate the Source

All paths below are relative to SGLang's `python/sglang/srt/speculative/`:

| File and entry point | What to look for |
| --- | --- |
| `standalone_worker_v2.py`: `StandaloneDraftWorker`, `StandaloneWorkerV2` | The independent draft model and vocabulary compatibility checks |
| `eagle_worker_v2.py`: `EAGLEWorkerV2.forward_batch_generation` | The order of drafting, target verification, and draft-state updates |
| `eagle_info.py`: `EagleVerifyInput` | Draft tokens, positions, retrieval indices, and verification metadata |

An independent draft reuses some EAGLE execution infrastructure, although their model mechanisms differ. First understand a single linear draft, then return to these entry points. Candidate trees and GPU kernels are beyond this walkthrough.

## 3 Connect Drafting to Verification

### 3.1 Call order in the source

The ordinary drafting path in `forward_batch_generation` can be reduced to this sequence:

```python
# eagle_worker_v2.py: EAGLEWorkerV2.forward_batch_generation (simplified)
verify_input = self.draft_worker.draft(batch)
batch.spec_info = verify_input
batch_output = self.verify(batch)
self.draft_worker._draft_extend_for_decode(batch, batch_output)
```

This omits parallel contexts, publication callbacks, and special branches. `draft` prepares candidates and verification metadata; `verify` calls the target and determines acceptance; the final call updates draft-side state for the next round. Following `verify` leads to `run_eagle_verify`.

### 3.2 Each linear proposal depends on earlier candidates

For the teaching interface below, `tokens` is a nonempty list of confirmed token IDs. `draft(prefix)` returns logits after each input position, shaped `[len(prefix),V]`. Target and draft must assign the same meaning to token IDs. `draft_steps` is a positive integer, `remaining` is the remaining output budget, and `eos_id` is the end-token ID (or `None` if unused). Start another round only while output budget remains. Greedy drafting is:

```python
# Teaching fragment: greedy linear drafting
proposals = []
for _ in range(min(draft_steps, remaining)):
    token = int(draft(tokens + proposals)[-1].argmax())
    proposals.append(token)
    if token == eos_id:
        break
```

Candidates stay in `proposals` until verified; they do not go straight into confirmed output. A proposed EOS stops further drafting, but the target must still verify it.

### 3.3 One target forward pass supplies k+1 rows

`target(prefix)` uses the same all-position logits interface. With confirmed prefix length `L` and draft length `k`, the alignment is:

| Zero-based target row | Purpose |
| --- | --- |
| `L-1` | Verify the first proposal |
| `L` | Verify the second proposal |
| `L+k-2` | Verify the last proposal |
| `L+k-1` | Choose an extra token, or bonus token, after full acceptance |

The teaching fragment selects these rows as follows:

```python
# Teaching fragment: full-prefix verification without a persistent KV cache
verification = target(tokens + proposals)[len(tokens) - 1:]
```

Row `i` predicts token `i+1`, so the slice starts at `L-1`. The causal mask prevents earlier prediction rows from seeing later proposals. A real engine can reuse cached prefixes; the full-prefix interface here makes alignment explicit.

## 4 Greedy Verification: Stop at the First Rejection

### 4.1 Commit the accepted prefix and correction token

This teaching function handles commitment for one round only. `verification` is the `[k+1,V]` tensor from the previous section, `remaining` is a nonnegative integer, and the return value is the list of tokens confirmed in this round.

```python
# Teaching fragment: greedy commitment for one verification round
def greedy_commit(proposals, verification, remaining, eos_id=None):
    if remaining == 0:
        return []
    output = []
    for i, proposed in enumerate(proposals):
        token = int(verification[i].argmax())
        output.append(token)
        if token == eos_id or len(output) == remaining:
            return output
        if token != proposed:
            return output
    output.append(int(verification[len(proposals)].argmax()))
    return output
```

If the target choice matches the proposal, verification continues. At the first mismatch, the function commits the target choice and returns. After full acceptance, the last row supplies a bonus token if output budget remains. The caller appends the returned list to the confirmed prefix and stops if the final token is EOS or the budget is exhausted, including when the bonus is EOS.

### 4.2 Work through one rejection

Suppose the draft proposes `[1,1,1]` and the target selects 1 and 2 in its first two rows. The first 1 is accepted, the second is corrected to 2, and the round commits only `[1,2]`. Remaining proposals were predicted under the wrong prefix. Discard them and the later verification rows.

The greedy correctness condition is that each committed token is the target's choice under the current confirmed prefix. With identical numerical computation and tie-breaking, the output should match target-only generation.

## 5 Random Verification and Cache Boundaries

### 5.1 Acceptance probability and the positive residual

In the random path, a candidate `y` drawn from draft distribution `q` is accepted with probability `min(1, p[y] / q[y])`. After rejection, a correction is sampled from the normalized positive residual. This teaching fragment assumes that `p` and `q` are one-dimensional probability tensors over the same vocabulary and conditional prefix, with `q[y] > 0`:

```python
import torch

acceptance = min(1.0, float(p[y] / q[y]))
if torch.rand(()).item() < acceptance:
    token = y
else:
    residual = (p - q).clamp_min(0)
    token = int(torch.multinomial(residual / residual.sum(), 1))
```

Here, `p` and `q` must be the actual sampling distributions. If temperature, Top-k, or Top-p is applied, retain the processed probabilities. In exact arithmetic, rejection implies positive residual mass; an implementation must also check numerical validity under finite precision.

For example, let `p=[0.4,0.6]` and `q=[0.8,0.2]`. Proposal 0 is accepted with probability 0.5, proposal 1 is always accepted, and rejection can only produce correction 1. The final probability of 0 is `0.8×0.5=0.4`, and that of 1 is `0.2+0.8×0.5=0.6`. Simply resampling from `p` after rejection would change this result.

Acceptance contributes `min(p(v),q(v))`; rejection fills the remaining positive difference. Using the correct conditional prefix at every step preserves the target sequence distribution. Stop the round at the first rejection, and sample any bonus from the target distribution. Random decoding preserves distributions, not token-by-token identity under the same seed.

### 5.2 Return to verification inputs and KV caching

In `EagleVerifyInput`, inspect `draft_token`, `positions`, `custom_mask`, and `retrieve_*`. Real verification records candidate positions and visibility, requiring more than concatenated token IDs. When rejection sampling is enabled, `draft_probs` stores the per-step draft distributions used during verification.

The teaching fragments do not maintain persistent KV. A real engine may already have cached rejected proposals and must restore the confirmed path after the first rejection. A correction or bonus token has just been selected and usually has not completed its own forward pass. Rejected candidates' KV cannot serve as its cache. As you follow execution, track accepted lengths, cache indices, and draft-state updates together.

Fewer target calls do not imply a proportional speedup. Benefits also depend on draft latency, accepted length, verification, and cache-management costs, and must be measured with real models and hardware.

## 6 Summary and Exercises

### 6.1 Summary

Speculative decoding separates proposals from confirmed output. Verification rows must align with the prefix before each candidate, and commitment retains only the accepted prefix and any required correction. Greedy decoding compares target choices; random decoding uses acceptance probabilities and positive residuals. EOS, output limits, and cache restoration must all follow the confirmed output.

### 6.2 Exercises

1. With prefix length 5 and draft length 3, which rows verify the proposals and select the bonus?
2. Why must later verification rows be discarded after the first rejection?
3. In Section 5.1's example, what output probabilities result from resampling directly from `p` after rejection?
4. If a proposed EOS is rejected, what should be committed, and should generation stop?
5. Why does equal vocabulary size not establish target/draft compatibility?
6. Why can a correction token not reuse the rejected candidate's KV?

## References

- [SGLang: Independent draft model](https://github.com/sgl-project/sglang/blob/1291f3175d3b7614821884e9653985fcd3c3dae8/python/sglang/srt/speculative/standalone_worker_v2.py)
- [SGLang: Speculative execution flow](https://github.com/sgl-project/sglang/blob/1291f3175d3b7614821884e9653985fcd3c3dae8/python/sglang/srt/speculative/eagle_worker_v2.py)
- [SGLang: Draft and verification inputs](https://github.com/sgl-project/sglang/blob/1291f3175d3b7614821884e9653985fcd3c3dae8/python/sglang/srt/speculative/eagle_info.py)
- [Leviathan et al.: Fast Inference from Transformers via Speculative Decoding, Algorithm 1](https://arxiv.org/abs/2211.17192)
