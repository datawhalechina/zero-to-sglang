# Chapter 6 Continuous Batching and the Scheduler (Code Walkthrough)

A and B are generating when C arrives. If A finishes first, how can C use its slot? Building on the [concept chapter](./Chapter6_Continuous-Batching-and-the-Scheduler.md), this walkthrough follows mini-sglang's receive, schedule, execute, and cleanup paths to see how batches change between steps.

The reference is pinned to commit [9a91cfa](https://github.com/sgl-project/mini-sglang/tree/9a91cfafe754aa85daee49998176275667eb58f2). Code blocks are source excerpts, with simplifications labeled explicitly. They depend on classes and runtime setup in the original files and are not standalone schedulers. Read the synchronous `normal_loop` before studying the additional state needed for overlapping execution.

## 1 Learning Objectives

After completing this chapter, you will be able to:

1. Locate the waiting queue, running set, and transitions between them.
2. Explain prefill priority, request slots, and token budgets from the source.
3. Trace output updates, EOS, length limits, and resource cleanup.

## 2 Locate the Source

All paths below are relative to mini-sglang's `python/minisgl/`:

| File and entry point | What to look for |
| --- | --- |
| `scheduler/scheduler.py`: `Scheduler.normal_loop` | Receiving, scheduling, computing, and processing results in one step |
| `scheduler/prefill.py`: `PrefillManager`, `PrefillAdder` | Admitting waiting requests and checking budgets |
| `scheduler/decode.py`: `DecodeManager` | Requests that continue generating |
| `core.py`: `Req` | Cached length, remaining output budget, and request state |
| `scheduler/table.py`, `scheduler/cache.py` | Request slots and cache resources |

After HTTP handling and tokenization, the scheduler receives a `UserMsg`. `_process_one_msg` checks the model's sequence-length limit, reduces the maximum output length if necessary, and calls `prefill_manager.add_one_req` to enqueue the request.

## 3 Read One Scheduling Step

### 3.1 Receive, schedule, execute, and process results

`normal_loop` runs the following sequence:

```python
# scheduler/scheduler.py: Scheduler.normal_loop
blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
for msg in self.receive_msg(blocking=blocking):
    self._process_one_msg(msg)

forward_input = self._schedule_next_batch()
ongoing_data = None
if forward_input is not None:
    ongoing_data = (forward_input, self._forward(forward_input))

self._process_last_data(ongoing_data)
```

When work is available, receiving messages does not block computation for the step. With no work, the loop waits for new messages. `ongoing_data` pairs the step's input and output so result processing can associate each token with its request.

### 3.2 Try prefill first, otherwise decode

`_schedule_next_batch` chooses a batch as follows:

```python
# scheduler/scheduler.py: Scheduler._schedule_next_batch
batch = (
    self.prefill_manager.schedule_next_batch(self.prefill_budget)
    or self.decode_manager.schedule_next_batch()
)
return self._prepare_batch(batch) if batch else None
```

The `or` expresses a policy: prefer a prefill batch if one can be formed; otherwise advance decode. **A nonempty waiting queue does not guarantee an admissible prefill batch.** Resource limits can force a decode step instead. `_prepare_batch` then allocates required KV pages, constructs positions and mappings, and prepares attention metadata.

Continuous batching does not require mixing new and existing requests into every batch. This version selects prefill or decode on different steps while allowing the active request set to change continuously.

## 4 How Requests Enter and Stay in a Batch

### 4.1 Try waiting requests in order

`PrefillManager.pending_list` stores requests in arrival order. This excerpt shows admission order only, omitting the bookkeeping for chunked prefill:

```python
# scheduler/prefill.py: PrefillManager.schedule_next_batch (simplified)
reqs = []
for pending_req in self.pending_list:
    if req := adder.try_add_one(pending_req):
        reqs.append(req)
    else:
        break
```

If the next request cannot enter, admission stops for that step rather than searching for a smaller request later in the queue. The full implementation also returns unfinished prefill requests to the front so their remaining prompts can continue on the next step.

### 4.2 Keep three resource limits distinct

`PrefillAdder` checks these constraints separately:

| Constraint | Source check | Meaning |
| --- | --- | --- |
| Request slots | `table_manager.available_size` | Whether a request-table row is available |
| Compute budget for this step | `token_budget` | How many prefill tokens this step can process |
| Estimated KV capacity | `estimated_len + reserved_size` | Token space reserved for new and running requests |

For initial admission, the capacity estimate centers on:

```python
# scheduler/prefill.py: PrefillAdder._try_allocate_one (excerpt)
extend_len = req.input_len - cached_len
estimated_len = extend_len + req.output_len
if estimated_len + self.reserved_size > self.cache_manager.available_size:
    return None
```

`cached_len` comes from prefix matching. `reserved_size` starts with `decode_manager.inflight_tokens` and grows as requests enter the current batch. Locking a cache handle can change reclaimable capacity, so the source checks capacity again after locking.

A long prompt need not run in one step: `_add_one_req` chooses a chunk with `min(token_budget, remain_len)`. If more remains, it creates a `ChunkedReq`. Intermediate chunks do not commit output tokens or enter the decode set. These budgets estimate token capacity, not memory usage in bytes.

### 4.3 Keep requests that still need to generate

After calling the engine, `_forward` updates the running set through `DecodeManager.filter_reqs`:

```python
# scheduler/decode.py: DecodeManager.filter_reqs
self.running_reqs = {
    req for req in self.running_reqs.union(reqs) if req.can_decode
}
```

`Req.can_decode` indicates remaining length budget. EOS is checked later when sampled tokens reach the host. The next decode batch sorts the running set by uid and advances each request by one token. Outputs remain associated with each request's uid, table row, and length.

## 5 Update Outputs and Change the Batch

### 5.1 Complete requests and release resources

`_process_last_data` first waits for tokens to be copied from GPU to CPU, then processes them in `batch.reqs` order. This excerpt omits reply messages, chunk checks, and the guard against duplicate cleanup during overlapping execution:

```python
# scheduler/scheduler.py: Scheduler._process_last_data (simplified)
next_token = next_tokens_cpu[i]
req.append_host(next_token.unsqueeze(0))
finished = not req.can_decode
if not req.sampling_params.ignore_eos:
    finished |= int(next_token.item()) == self.eos_token_id
if finished:
    self.decode_manager.remove_req(req)
    self._free_req_resources(req)
```

`_free_req_resources` returns the request-table row and calls `cache_manager.cache_req(req, finished=True)`. Completion does not immediately delete all physical KV: reusable prefixes may remain cached. Chapters 7 and 8 explain that lifecycle.

Cancellation uses the `AbortBackendMsg` branch of `_process_one_msg`: it removes the request from the waiting queue or running set, then calls the same cleanup entry point if resources were allocated. A queued request with no resources only needs removal from the queue. Overlapping execution also has computation in flight, so inspect synchronization and duplicate-cleanup guards alongside this path.

### 5.2 Work through A, B, and C

Assume two request slots, sufficient KV capacity, no prefix-cache hits, and prompts that each finish prefill in one step. A, B, and C need 2, 5, and 2 output tokens respectively, with no early EOS. C arrives after step 0. Finishing prefill produces the first output token.

| Step | Batch | State at the end of the step |
| --- | --- | --- |
| 0 | Prefill: A, B | A and B receive their first token; C then arrives |
| 1 | Decode: A, B | A finishes after its second token, releasing a slot |
| 2 | Prefill: C | C uses the slot; B pauses for this step |
| 3 | Decode: B, C | C finishes after its second token |
| 4 | Decode: B | B receives its fourth token |
| 5 | Decode: B | B finishes after its fifth token |

C can enter before B finishes: this is continuous batching in this example. The table is a trace derived from the stated assumptions, not a server log or timing measurement. Persistent prefill priority can delay existing requests' decode steps, so fairness must be evaluated with real workloads.

## 6 Summary and Exercises

### 6.1 Summary

`normal_loop` connects receiving, scheduling, execution, and result processing. PrefillManager decides who can enter, DecodeManager retains requests that continue generating, and result processing handles completion and cleanup. Continuous batching updates these states each step so new requests can reuse available capacity.

### 6.2 Exercises

1. Why might decode run even when the waiting queue is nonempty?
2. Can a request enter with an available request slot but insufficient KV capacity? Which code checks this?
3. Why must intermediate prefill chunks not commit generated tokens?
4. Under the assumptions in Section 5.2, when can C first enter if A generates only one token?
5. Where do completion and cancellation enter cleanup, and why does releasing a request slot not delete all KV?

## References

- [mini-sglang: Scheduling loop](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/scheduler.py)
- [mini-sglang: Prefill admission](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/prefill.py)
- [mini-sglang: Decode running set](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/decode.py)
- [mini-sglang: Request state](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/core.py)
- [mini-sglang: Request table](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/table.py)
- [mini-sglang: Cache management](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/cache.py)
- [Chapter 7: Paged KV Cache and Memory Management](./Chapter7_Paged-KV-Cache-and-Memory-Management.md)
- [Chapter 8: RadixAttention and Prefix Caching](./Chapter8_RadixAttention-and-Prefix-Caching.md)
