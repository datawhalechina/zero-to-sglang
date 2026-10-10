"""Chapter 6: FIFO admission, prefill priority, cached decode and request cleanup.

The token reservation is accounting, not a physical GPU page allocator.
The eager backend packs/copies per-request KV tensors to keep ownership visible.
"""

from collections import deque
from dataclasses import dataclass, field

import torch

from .forward import KVCache, Qwen3


@dataclass
class Request:
    uid: str
    prompt: list[int]
    max_new_tokens: int
    eos_id: int | None = None
    output: list[int] = field(default_factory=list)
    state: str = "waiting"
    finish_reason: str | None = None
    cache: KVCache | None = None
    first_token_tick: int | None = None
    finish_tick: int | None = None

    @property
    def reservation(self):
        return len(self.prompt) + self.max_new_tokens


class KVEngine:
    """Exactly one batched model forward per call. Sampling is greedy."""

    def __init__(self, model):
        self.model = model.eval()
        self.calls = []

    @torch.inference_mode()
    def step(self, requests, phase):
        device = next(self.model.parameters()).device
        batch = len(requests)
        if phase == "prefill":
            lengths = [len(r.prompt) for r in requests]
            width = max(lengths)
            ids = torch.zeros((batch, width), dtype=torch.long, device=device)
            mask = torch.zeros_like(ids, dtype=torch.bool)
            for i, r in enumerate(requests):
                ids[i, :lengths[i]] = torch.tensor(r.prompt, device=device)
                mask[i, :lengths[i]] = True
            past = None
            last = torch.tensor(lengths, device=device) - 1
        elif phase == "decode":
            # Cache contains the prefix BEFORE the last sampled token.
            lengths = [r.cache[0][0].shape[2] for r in requests]
            width = max(lengths)
            ids = torch.tensor([[r.output[-1]] for r in requests], device=device)
            mask = torch.zeros((batch, width + 1), dtype=torch.bool, device=device)
            mask[:, -1] = True
            past = []
            for layer in range(len(self.model.model.layers)):
                sample = requests[0].cache[layer][0]
                shape = (batch, sample.shape[1], width, sample.shape[3])
                k, v = sample.new_zeros(shape), sample.new_zeros(shape)
                for i, r in enumerate(requests):
                    k[i:i+1, :, :lengths[i]] = r.cache[layer][0]
                    v[i:i+1, :, :lengths[i]] = r.cache[layer][1]
                    mask[i, :lengths[i]] = True
                past.append((k, v))
            last = torch.zeros(batch, dtype=torch.long, device=device)
        else:
            raise ValueError("phase must be prefill or decode")
        logits, updated = self.model(ids, attention_mask=mask, cache=past, use_cache=True)
        next_ids = logits[torch.arange(batch, device=device), last].argmax(-1).tolist()
        for i, r in enumerate(requests):
            valid = mask[i].nonzero().flatten()
            r.cache = [(k[i:i+1].index_select(2, valid), v[i:i+1].index_select(2, valid))
                       for k, v in updated]
        self.calls.append((phase, tuple(r.uid for r in requests), tuple(ids.shape)))
        return next_ids


class Scheduler:
    def __init__(self, engine, max_running=2, token_capacity=256, prefill_budget=64):
        if min(max_running, token_capacity, prefill_budget) <= 0:
            raise ValueError("all scheduler capacities must be positive")
        self.engine, self.max_running = engine, max_running
        self.token_capacity, self.prefill_budget = token_capacity, prefill_budget
        self.waiting, self.running, self.requests = deque(), {}, {}
        self.reserved, self.tick = 0, 0
        self.trace = []

    def submit(self, uid, prompt, max_new_tokens, eos_id=None):
        if uid in self.requests:
            raise ValueError("request uid must be unique")
        if not prompt or max_new_tokens < 0:
            raise ValueError("nonempty prompt and nonnegative output length required")
        c = self.engine.model.config
        if any(t < 0 or t >= c.vocab_size for t in prompt):
            raise ValueError("token id outside vocabulary")
        request = Request(uid, list(prompt), max_new_tokens, eos_id)
        if request.reservation > min(c.max_position_embeddings, self.token_capacity):
            raise ValueError("request can never fit context/token capacity")
        if len(prompt) > self.prefill_budget:
            raise ValueError("prompt exceeds prefill budget; chunked prefill is not implemented")
        self.requests[uid] = request
        if max_new_tokens == 0:
            self._finish(request, "length")
        else:
            self.waiting.append(request)
        return request

    def _finish(self, request, reason):
        if request.state in {"finished", "cancelled"}:
            return
        if request.uid in self.running:
            del self.running[request.uid]
            self.reserved -= request.reservation
        request.cache = None
        request.state = "cancelled" if reason == "cancelled" else "finished"
        request.finish_reason, request.finish_tick = reason, self.tick

    def cancel(self, uid):
        request = self.requests.get(uid)
        if request is None or request.state in {"finished", "cancelled"}:
            return False
        self.waiting = deque(r for r in self.waiting if r.uid != uid)
        self._finish(request, "cancelled")
        return True

    @property
    def has_work(self):
        return bool(self.waiting or self.running)

    def step(self):
        admitted, budget = [], self.prefill_budget
        while self.waiting and len(self.running) < self.max_running:
            request = self.waiting[0]
            if len(request.prompt) > budget or self.reserved + request.reservation > self.token_capacity:
                break  # Strict FIFO: no bypassing the first waiting request.
            self.waiting.popleft()
            request.state = "running"
            self.running[request.uid] = request
            self.reserved += request.reservation
            budget -= len(request.prompt)
            admitted.append(request)
        phase = "prefill" if admitted else "decode"
        batch = admitted or list(self.running.values())
        if not batch:
            return []
        # Existing decodes pause during a prefill iteration, matching the pinned policy.
        try:
            next_ids = self.engine.step(batch, phase)
            if len(next_ids) != len(batch):
                raise RuntimeError("engine must return one token per request")
        except Exception:
            for request in batch:
                self._finish(request, "error")
            raise
        events = []
        for request, token in zip(batch, next_ids):
            request.output.append(token)
            if request.first_token_tick is None:
                request.first_token_tick = self.tick
            if token == request.eos_id:
                self._finish(request, "eos")
            elif len(request.output) == request.max_new_tokens:
                self._finish(request, "length")
            events.append((request.uid, token, request.finish_reason))
        self.trace.append({"tick": self.tick, "phase": phase,
                           "batch": [r.uid for r in batch], "events": events,
                           "reserved_after": self.reserved})
        self.tick += 1
        assert self.reserved == sum(r.reservation for r in self.running.values())
        return events


def main():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    scheduler = Scheduler(KVEngine(Qwen3()))
    scheduler.submit("A", [1, 2], 2)
    scheduler.submit("B", [3, 4, 5], 5)
    scheduler.step()
    scheduler.submit("C", [6], 2)  # Arrives while A/B are already generating.
    while scheduler.has_work:
        scheduler.step()
    for row in scheduler.trace:
        print(f"tick={row['tick']} {row['phase']:7} batch={row['batch']} "
              f"reserved_after={row['reserved_after']}")
    print("All request caches released:", all(r.cache is None for r in scheduler.requests.values()))


if __name__ == "__main__":
    main()
