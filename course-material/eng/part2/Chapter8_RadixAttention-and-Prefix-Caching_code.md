# Chapter 8 RadixAttention and Prefix Caching (code)

The previous chapter introduced the Radix Tree, longest common prefixes, the KV Cache Pool, and the cache lifecycle $match \to lock \to allocate \to insert \to unlock \to evict/free$. This chapter applies those abstractions to the mini-SGLang source. We follow a request that hits the cache from prefix matching onward, observing how indices are produced, physical pages are allocated, KV is written, and resources are reclaimed when the request ends.

Keep the following idea as the guiding thread: “The Radix Tree records mappings, while the KV Cache Pool stores the data.” We recommend reading the [conceptual chapter](./Chapter 8_RadixAttention and Prefix Caching.md) first, then using this chapter as a code companion.

## 1 Learning Objectives

After completing this chapter, you should be able to answer the following questions clearly:

- Locate the implementations of Prefix Cache, page table, and KV Cache Pool in mini-sglang, and explain how the three components work together.
- Explain how a request’s complete lifecycle, from prefix matching to completion, is connected in the code.
- Explain what each core Prefix Cache operation does.


## 2 Locating the source code

The code discussed in this chapter is located under **python/minisgl/**:

| File | Purpose |
|------|------|
| kvcache/base.py | Defines the abstract interfaces for Prefix Cache and KV Cache Pool |
| kvcache/radix_cache.py | Implements the Radix Tree, cache handles, matching, insertion, locking, and eviction of evictable leaves |
| scheduler/cache.py | Connects the Radix Tree, free_slots, and the page table |
| scheduler/prefill.py | Matches prefixes before batching, locks handles, writes the matched segment, and performs chunked prefill |
| scheduler/table.py | Allocates and releases request slots and rows in token_pool / page table |
| scheduler/decode.py | Manages the Decode run queue and estimates inflight_tokens |
| scheduler/scheduler.py | Builds batches, allocates new pages, prepares Attention inputs, and reclaims resources when requests finish or are cancelled |
| kvcache/mha_pool.py | Stores K and V tensors layer by layer on the GPU |
| kernel/csrc/jit/store.cu | CUDA implementation for writing to the KV buffer |
| kernel/csrc/src/radix.cpp | Accelerates key comparisons in the Radix Tree |
| kernel/radix.py | Python wrapper around the radix C++ implementation above |

We first identify each file’s responsibility, then connect them along the request lifecycle. When reading the code below, ask of each function: is it maintaining a mapping, or operating on physical KV data?

## 3 Distinguishing the mapping table from the actual KV storage

In the RadixAttention code, the easiest thing to confuse is the **mapping relationship** and the **actual KV Cache Pool**. Token-id sequences and physical slots were covered in the preceding conceptual chapter, so here we focus on the two components:

| Component | Responsibility | What it stores |
|------|------|--------|
| **Radix Tree** | Prefix matching and indexing | A mapping from **token-id sequence → physical-slot indices** (along with tree topology, ref_count, and so on) |
| **MHAKVCache** | Physical KV storage pool | Contiguous K/V Tensor buffers (paged layout) |

The Radix Tree **does not store any KV floating-point data**. It only records which physical-buffer slots hold the KV for a particular token segment.

The actual KV Tensor is allocated in **MHAKVCache**. After layers/attention.py computes it, a CUDA kernel (**store.cu → store_cache**) writes it to the slots specified by the indices.

**1. Physical storage: MHAKVCache**

```python
# python/minisgl/kvcache/mha_pool.py
class MHAKVCache(BaseKVCachePool):
    def __init__(self, num_kv_heads, num_layers, head_dim,
                 num_pages, page_size, dtype, device):
        ...
        self._kv_buffer = torch.empty(
            (2, num_layers, num_pages, page_size, local_kv_heads, head_dim),
            device=device, dtype=dtype,
        )
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)

    def store_kv(self, k, v, out_loc, layer_id):
        from minisgl.kernel import store_cache
        store_cache(
            k_cache=self._k_buffer[layer_id].view(self._storage_shape),
            v_cache=self._v_buffer[layer_id].view(self._storage_shape),
            indices=out_loc,  # Preallocated physical slot indices
            k=k, v=v,
        )
```

- **_kv_buffer** accounts for most of the memory used by the **KV payload** (paged layout).
- **out_loc** (indices) comes from the scheduling stage (the page table and related structures) and tells the kernel which physical slots should receive the KV computed for the current step.


**2. Mapping table: RadixTreeNode**

```python
# python/minisgl/kvcache/radix_cache.py
class RadixTreeNode:
    def set_key_value(self, key: torch.Tensor, value: torch.Tensor):
        self._key = key
        self._value = value
        self._length = len(key)

class RadixCacheHandle:
    def get_matched_indices(self) -> torch.Tensor:
        # Walk from the current node to the root through parent, concatenating node.value along the way
        # Return the complete list of physical slots for attention to use
        ...
```

- Each node stores only a segment of contiguous token **indices**; it does not store float16/bfloat16 K/V data.
- **match_prefix** matches the longest prefix in the tree by token sequence. When necessary, it calls **split_at** to adjust the topology and returns a handle containing **prefix_len** and the node.
- **insert_prefix** registers **(token_ids, page_indices)** in the tree after computation and **store_kv** complete, making them available for later requests.
- The actual KV storage always remains in **MHAKVCache**. The Radix Tree only records the mapping from this token-id segment to its physical slots.

**Note:** When the cache hits, attention only needs the indices to read existing data. Only the cache-miss portion is computed and passed to **store_kv**, after which **insert_prefix** is called at the appropriate time.

In one sentence, the core relationship is: **Radix Tree = index (token id → physical slots); MHAKVCache = the GPU memory pool that actually stores KV**.

Once the relationship between the index and the physical pool is clear, read the following code through one question: where do the indices for this token segment come from, when are they protected, and who releases them at the end? We now trace a complete cache-hit request to connect the components.

## 4 Walking through the source code for a cache hit

We follow the call order: create the cache objects, use match_prefix to find and lock the matched path, reserve and allocate pages for the cache-miss portion, write physical KV with store_kv during forward, and finally register the reusable indices and corresponding token-id sequence back into the Radix Tree with insert_prefix.

---

### 4.1 Path-compressed nodes: RadixTreeNode

First, look at how the cache index itself is organized. mini-SGLang provides two prefix-cache strategies: naive does not reuse prefixes across requests, while radix uses a path-compressed Radix Tree to share physical-page indices by token-id prefix. Matching, locking, and eviction all build on this tree:

```python
# kvcache/__init__.py (excerpt)
@SUPPORTED_CACHE_MANAGER.register("radix")
def create_radix_cache(device: torch.device):
    return RadixPrefixCache(device=device)

def create_prefix_cache(device: torch.device, type: str) -> BasePrefixCache:
    return SUPPORTED_CACHE_MANAGER[type](device)
```

Here we choose **radix**. The Scheduler initializes **CacheManager** to provide the foundation for subsequent request processing. It holds the **Prefix Cache** (RadixPrefixCache), the **free-page list** (free_slots), and the **global page table** (the slot mapping used while processing requests).


A node in the Radix Tree represents **a contiguous token sequence** (which may span multiple pages), rather than a single token:

```python
# kvcache/radix_cache.py (excerpt)
class RadixTreeNode:
    def __init__(self, key_fn, tic=None):
        self.children = {}
        self._parent = None
        self.ref_count = 0
        self.timestamp = tic or time.monotonic_ns()

    def set_key_value(self, key, value):
        assert len(key) == len(value)
        self._key = key      # token-id segment
        self._value = value  # corresponding physical-slot indices
        self._length = len(key)
```

When matching stops in the middle of a node, the common prefix must be split out at the branching point (triggered in _tree_walk):

```python
def split_at(self, pos):
    parent = self.parent
    new_node = RadixTreeNode(self.key_fn, self.timestamp)
    new_node.set_key_value(self._key[:pos], self._value[:pos])
    new_node.set_parent(parent)
    new_node.ref_count = self.ref_count   # Inherit the reference count

    self.set_key_value(self._key[pos:], self._value[pos:])
    self.set_parent(new_node)
    return new_node
```

For example, suppose the original node has key [10,11,12,13] and this request matches only [10,11]. After split_at(2), the new node holds the common prefix [10,11], while the original node is shortened to the suffix [12,13] and attached below the new node. The physical KV is not moved; only the indices are split. Because the common segment is still protected by the existing reference, the new node inherits ref_count.

At this point, we know how tree nodes store and split indices. The next question is how a new input_ids sequence walks this tree to find the longest reusable prefix.

---

### 4.2 Longest-prefix matching: deriving the reusable range from input_ids

Once the tree is initialized, matching is the first step for a request: find the longest reusable prefix in the Radix Tree and return a **handle**. The handle carries both `cached_len` and the matched node; the page table fill and lock operations below depend on it.

The key used by the child dictionary is determined by `page_size`: when `page_size=1`, it uses the next token id; when `page_size>1`, it uses the tuple of token ids in the next complete page:

```python
def _get_key_fn(page_size):
    if page_size == 1:
        return lambda x: x[0].item()
    return lambda x: tuple(x[:page_size].tolist())
```

Longest-prefix matching is performed in _tree_walk:

```python
def _tree_walk(self, input_ids):
    prefix_len = 0
    node = self.root_node
    tic = time.monotonic_ns()

    while prefix_len < len(input_ids):
        child = node.children.get(self.key_fn(input_ids[prefix_len:]))
        if child is None:
            return node, prefix_len
        node = child

        match_len = node.get_match_len(input_ids[prefix_len:])
        match_len = align_down(match_len, self.page_size)
        prefix_len += match_len

        if match_len != node.length:  # Stop in the middle of a node → split it
            node = node.split_at(match_len)
            node.timestamp = tic
            return node, prefix_len

        node.timestamp = tic  # Full-node hit: refresh the LRU timestamp

    return node, prefix_len
```

Key points:

- **get_match_len**: compares the node’s _key with the remaining input_ids using fast_compare_key and returns the length of the matching segment.
- The match length is rounded down to a multiple of **page_size**, so a non-full-page prefix is not shared. Nodes on the matched path update their **timestamp**, so eviction prefers leaves with older use times.
- **split_at** is called only when matching stops in the middle of a node (match_len != node.length), and the split node is then returned.

match_prefix wraps the result in a handle:

```python
def match_prefix(self, input_ids):
    node, prefix_len = self._tree_walk(input_ids)
    return MatchResult(RadixCacheHandle(prefix_len, node))  # cached_len = prefix_len
```

When the matched segment must be written into the request’s page table, call **handle.get_matched_indices()**: walk from the current node to the root through parent, collect each segment’s value, reverse the list, and concatenate it with torch.cat to obtain physical indices in sequence order.

**Note:** Even when the tree contains no reusable prefix, match_prefix is still called and `prefix_len == 0`. This does not mean the request is skipped: after pages are allocated and forward plus store_kv complete for the cache-miss portion, insert_prefix registers it in the Radix Tree.

Matching only tells us “which content can be reused”; it has not prepared writable space for this request. The next step therefore moves from the read-only match result into scheduling: lock the matched prefix and reserve resources for the cache-miss suffix.

---

### 4.3 From a match result to an executable Batch: allocating pages and locking

After matching, the Scheduler has a **handle**. The matched prefix is still only a reusable range, so it must be locked to prevent eviction during this scheduling round. The scheduler also estimates the space required by the cache-miss suffix and the subsequent Decode phase. Physical pages are allocated only after the Batch is finalized, just before forward.

During Prefill, mini-SGLang uses a first-come, first-served strategy: it tries to add requests in **pending_list** order and `break`s as soon as space is insufficient, keeping kernel execution predictable:

```python
# scheduler/prefill.py (excerpt)
def try_add_one(self, pending_req: PendingReq) -> Req | None:
    if self.token_budget <= 0:
        return None
    # Existing chunked progress: continue with the next chunk
    if chunked_req := pending_req.chunked_req:
        return self._add_one_req(...)
    # New request: try allocation before officially adding it
    if resource := self._try_allocate_one(pending_req):
        ...
    return None

def schedule_next_batch(self, prefill_budget: int) -> Batch | None:
    if len(self.pending_list) == 0:
        return None
    adder = PrefillAdder(
        token_budget=prefill_budget,
        reserved_size=self.decode_manager.inflight_tokens,  # Estimated space reserved by Decode
        ...
    )
    reqs: List[Req] = []
    chunked_list: List[PendingReq] = []
    for pending_req in self.pending_list:
        if req := adder.try_add_one(pending_req):
            pending_req.chunked_req = None
            if isinstance(req, ChunkedReq):
                # Partially completed Req
                ...
            reqs.append(req) # Normal Req
        else:
            break  # Insufficient space; stop adding requests
    if len(reqs) == 0:
        return None
    # Put unfinished chunked requests back at the front to avoid starvation
    self.pending_list = chunked_list + self.pending_list[len(reqs):]
    return Batch(reqs=reqs, phase="prefill")
```

---

For a new request, **PrefillAdder._try_allocate_one** first estimates space and acquires the lock, reserving capacity for this Prefill extension and subsequent Decode:

```python
# scheduler/prefill.py (excerpt)
def _try_allocate_one(self, req: PendingReq) -> Tuple[BaseCacheHandle, int] | None:
    if self.table_manager.available_size == 0:
        return None

    # match_req matches input_ids[:input_len-1] (intentionally leaving the final token for the later extend)
    handle = self.cache_manager.match_req(req).cuda_handle
    cached_len = handle.cached_len
    extend_len = req.input_len - cached_len
    estimated_len = extend_len + req.output_len

    # First check: is there enough space for the Prefill extension and Decode reservation?
    if estimated_len + self.reserved_size > self.cache_manager.available_size:
        return None
    self.cache_manager.lock(handle)

    # Second check: After hitting the prefix lock, does it result in insufficient available preallocated space
    if estimated_len + self.reserved_size > self.cache_manager.available_size:
        return self.cache_manager.unlock(handle)  # unlock returns None

    table_idx = self.table_manager.allocate()
    if cached_len > 0:
        # Copy both the matched token ids and their physical page-table indices
        ...
    return handle, table_idx
```

**The two checks happen before and after locking: the first tests whether current resources are sufficient, and the second confirms that the state change caused by locking has not invalidated the allocation budget.** With that in mind, `available_size` is easier to understand: it is the number of tokens currently available from evicted prefix-cache entries and free pages:

```python
@property
def available_size(self) -> int:
    return self.prefix_cache.size_info.evictable_size + len(self.free_slots) * self.page_size
```

After the reservation succeeds, **PrefillAdder._add_one_req** formally submits the portion to compute in this round. It writes only token ids and chunks the request according to the budget; **physical pages have not been allocated yet**.

```python
# scheduler/prefill.py (excerpt)
def _add_one_req(
    self,
    pending_req: PendingReq,
    cache_handle: BaseCacheHandle,
    table_idx: int,
    cached_len: int,
) -> Req:
    remain_len = pending_req.input_len - cached_len
    chunk_size = min(self.token_budget, remain_len)
    is_chunked = chunk_size < remain_len
    CLS = ChunkedReq if is_chunked else Req
    self.token_budget -= chunk_size
    self.reserved_size += remain_len + pending_req.output_len

    # Update only token ids; the scheduler allocates physical pages when it calls allocate_paged
    _slice = slice(cached_len, cached_len + chunk_size)
    device_ids = self.table_manager.token_pool[table_idx, _slice]
    device_ids.copy_(...)
    return CLS(...)
```

To prevent a very long request from causing a high peak in GPU memory during one Prefill (OOM), the request is split into **chunks**. Each round processes only one chunk, which can be batched with shorter requests, instead of computing all chunks of the same request in a single round.

If the request is still `ChunkedReq == True` in this round, **schedule_next_batch()** puts it back at the front of **pending_list** so it continues first in the next round and does not starve. At this point we have only the request’s logical range and resource budget; the new pages have not yet been written into the page table.

---

When the Batch is assembled and Prefill forward is about to run, the scheduler turns the budget into a physical mapping by calling **allocate_paged**. It **allocates new physical pages only for the cache-miss interval**:

```python
# scheduler/cache.py (excerpt)
    def _page_to_token(self, pages: torch.Tensor) -> torch.Tensor:
        if self.page_size == 1:
            return pages
        # [X * page_size] -> [X * page_size, ..., X * page_size + page_size - 1]
        ...
        return (pages.unsqueeze(1) + offsets).flatten()

def allocate_paged(self, reqs: List[Req]) -> None:
    needed_pages = 0
    allocation_info: List[Tuple[int, int, int]] = []
    for req in reqs:
        first_page = div_ceil(req.cached_len, self.page_size)
        last_page = div_ceil(req.device_len, self.page_size)
        if last_page > first_page:
            needed_pages += last_page - first_page
            allocation_info.append((req.table_idx, first_page, last_page))
    if needed_pages > 0:
        allocated = self._page_to_token(self._allocate(needed_pages))
        _write_page_table(self.page_table, allocated, allocation_info, self.page_size)
```

The page table for [0, cached_len) was filled on a cache hit by **handle.get_matched_indices()**. Starting at ceil(cached_len / page_size), this step allocates space only for the new pages covering [cached_len, device_len). **_page_to_token** expands each page’s starting slot into complete token indices before writing them into the page table.

If the free list is insufficient, **_allocate** first evicts leaf nodes from the Radix Tree as needed, obtains their token slots, returns each page’s starting slot to **free_slots**, and then completes the allocation:

```python
# scheduler/cache.py (excerpt)
def _allocate(self, needed_pages: int) -> torch.Tensor:
    if needed_pages > (free_pages := len(self.free_slots)):
        evicted = self.prefix_cache.evict((needed_pages - free_pages) * self.page_size)
        self.free_slots = torch.cat([self.free_slots, evicted[:: self.page_size]])
        ...
    allocated = self.free_slots[:needed_pages]
    self.free_slots = self.free_slots[needed_pages:]
    return allocated
```

**Note:** **RadixPrefixCache.evict** only detaches nodes from the tree and returns their physical indices. **CacheManager._allocate** (together with _free and lazy_free_region) is responsible for actually returning pages to the allocator. **free_slots** always contains page-aligned starting token indices.

The cache-miss suffix now has writable physical locations. Only after forward completes do the KV values in those locations become eligible for reuse. The next step is therefore to see how `cache_req` writes them back to the prefix cache while handling both release and continued ownership.

---

After the Prefill forward finishes, **cache_req** writes the result back to the prefix cache and releases pages that are no longer needed, while protecting prefixes that are still in use:

```python
# scheduler/cache.py (excerpt)
def cache_req(self, req: Req, *, finished: bool) -> None:
    insert_ids = req.input_ids[: req.cached_len]
    page_indices = self.page_table[req.table_idx, : req.cached_len]
    old_handle = req.cache_handle
    cached_len, new_handle = self.prefix_cache.insert_prefix(insert_ids, page_indices)
    self.unlock(old_handle)
    self._free(page_indices[old_handle.cached_len : cached_len])  # Portion already written to the prefix cache by another request; release it to prevent a leak
    if finished:
        self._free(page_indices[new_handle.cached_len :])         # Request finished; release the tail
    else:
        req.cache_handle = new_handle
        self.lock(new_handle)                                     # Keep it protected for Decode
```

The integrity check is used to detect page leaks early and limit their impact:

```python
def check_integrity(self) -> None:
    self.prefix_cache.check_integrity()
    cache_pages = self.prefix_cache.size_info.total_size // self.page_size
    if len(self.free_slots) + cache_pages != self.num_pages:
        raise RuntimeError(...)
    if self.page_size > 1:
        assert torch.all(self.free_slots % self.page_size == 0)
```

This completes archiving the Prefill result, but whether the request can enter Decode still depends on whether it is a complete Req. This boundary matters because ChunkedReq uses it to prevent a partially completed Prefill from starting sampling.

---

After Prefill is fully complete, the request becomes eligible for Decode. ChunkedReq overrides `can_decode` to always return False, so it is not added to the Decode manager; it represents a Prefill still being processed in chunks and cannot sample early. **Only a normal Req (Prefill complete) with `remain_len > 0`** enters Decode. In other words, for an individual request, Decode starts when Prefill is complete, not merely when one chunk has finished.

After each Decode forward pass, the Engine calls **req.complete_one()**:

```python
# core.py
def complete_one(self) -> None:
    self.cached_len = self.device_len
    self.device_len += 1
```

The newly generated token is added to the cached length, and space is reserved in device_len for the next token.

---

### 4.4 Decode, normal completion, and cancellation

Once Decode begins, the role of the **handle** also changes: it continues to lock the prefix archived at the end of Prefill, while new suffix KV produced during intermediate steps is written only to the request’s own page-table slots. Decode **does not** insert the new suffix into the Radix Tree after every step; the Engine calls `req.complete_one()` to advance the lengths, and the Scheduler uses `append_host` to append the new token to `input_ids`. Only when the request ends does the system decide which suffixes are worth archiving.

When a request **finishes normally** (`remain_len == 0` or EOS is reached), resources are released through a single path:

```python
# scheduler/scheduler.py (excerpt)
def _free_req_resources(self, req: Req) -> None:
    self.table_manager.free(req.table_idx)
    self.cache_manager.cache_req(req, finished=True)
```

With cache_req(..., finished=True), input_ids[:cached_len] and the corresponding page indices are passed to insert_prefix: the page-aligned portion that can enter the tree is retained as reusable cache; intermediate segments already occupied by other requests and the unaligned tail page are returned to the free list with **_free**.

When the **user cancels** a request, the Scheduler first searches the prefill and decode queues:

```python
# scheduler/scheduler.py (excerpt)
elif isinstance(msg, AbortBackendMsg):
    req_to_free = self.prefill_manager.abort_req(msg.uid)
    req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
    if req_to_free is not None:
        self._free_req_resources(req_to_free)
```

Normal completion and cancellation share **_free_req_resources()**, so both exit paths converge on the same cache-cleanup logic.

When processing the previous batch’s results, the release operations are also wrapped in **lazy_free_region**: multiple **_free** calls from the batch are held temporarily and concatenated back into free_slots once the context exits. This reduces intermediate state changes and keeps the batch update consistent:

```python
# scheduler/scheduler.py (excerpt)
with self.cache_manager.lazy_free_region():
    for req in batch.reqs:
        if isinstance(req, ChunkedReq):
            continue  # Partially completed Prefill: no sampling and no finalization here
        req.append_host(next_token)
        finished = (not req.can_decode) or (next_token == eos ...)
        if finished and req not in self.finished_reqs:
            self.decode_manager.remove_req(req)
            self._free_req_resources(req)          # cache_req(finished=True)
            new_finished_reqs.add(req)
        elif batch.is_prefill:
            # Non-chunked Prefill complete: write to the prefix cache and continue to Decode
            self.cache_manager.cache_req(req, finished=False)
```

This reduces repeated concatenation of small tensors and avoids repeatedly changing resource state while overlap scheduling is active. **finished_reqs** prevents the same request from being freed twice.

## 5 Comparing with real SGLang

The preceding flow showed how a request in mini-sglang completes matching, allocation, writing, and release. Real SGLang divides these responsibilities across more modules, but the same actions still correspond to one another:

| Feature | mini-sglang (`python/minisgl/`) | SGLang (`python/sglang/`) |
|------|----------------------------------|---------------------------|
| Radix / Prefix Cache | kvcache/radix_cache.py | srt/mem_cache/radix_cache.py; converging toward unified_cache/ (Unified Radix Cache), with variants such as HiCache |
| Request-side cache coordination (match / lock / allocation / release) | scheduler/cache.py | Split across srt/mem_cache/ and srt/managers/scheduler.py; much of the Prefill logic is in srt/managers/schedule_policy.py |
| Scheduling policy | Simplified FCFS: pending_list order + break when space is insufficient | srt/managers/schedule_policy.py |
| Kernel | kernel/ | Unified entry points under kernels/ (ops/ operators, jit/ compilation runtime, and so on); precompiled implementations also commonly use the separate sgl-kernel package |
| KV physical storage / allocation | kvcache/mha_pool.py and related files | srt/mem_cache/pool/ and srt/mem_cache/storage/ |

When reading SGLang, you can still follow the same lifecycle:

- **Where is the match result produced?** — The prefix tree / Unified Radix (mem_cache);
- **When is it locked?** — Scheduler coordination and lock_ref / handle protection;
- **Where is the cache-miss portion allocated?** — allocator + pool, corresponding to allocate_paged / free list in mini-SGLang;
- **Who writes and releases data on completion or cancellation?** — The cache insert / free path in the scheduler, corresponding to cache_req / _free_req_resources in mini-SGLang.

The files, storage media, and policies are more complex, but the main lifecycle is shared with mini-sglang. When reading real SGLang, first use mini-sglang to establish this line, then map each action to the more fine-grained modules.

## 6 Summary and Exercises


### 6.1 Summary

A Prefix Cache hit is not simply a matter of retrieving KV tensors from the Radix Tree. In mini-sglang:

1. **Radix Tree** matches token prefixes and produces **physical-page indices** written into the page table; the actual KV remains in the KV Cache pool;
2. A **cache handle + lock** protects nodes still used by active requests from eviction;
3. The Scheduler combines the matched indices and the **newly allocated** cache-miss suffix in one page table;
4. When the free list cannot allocate the cache-miss suffix, eviction starts from **evictable leaves**, and **CacheManager** returns the pages to free_slots;
5. When Prefill or the request ends, **cache_req** writes the prefix that can enter the tree back to the Radix Tree; intermediate segments already occupied by other requests and the unaligned tail are released with _free.

---

### 6.2 Exercises


1.Why does **CacheManager.match_req** receive only **input_ids[:input_len - 1]**? If the entire prompt were allowed to hit, what problem would the model encounter during the forward pass?
> Hint: Prefill must both write the KV Cache computed in this round and produce the **logits for the first output token**.


2.Why does **PrefillAdder._try_allocate_one** check available_size both before and after lock(handle)?
> Hint: both checks compare estimated_len + reserved_size (including the Decode reservation and the space reserved for Decode) with available_size; lock changes the evictable size in between.


3.Why does **RadixCacheHandle.get_matched_indices** walk through parent to the root and then reverse-concatenate the segments?

> Hint: each tree node stores only the value for its own segment, so start at the matched node and walk upward through parent.


4. In cache_req, what scenario does **page_indices[old_handle.cached_len:cached_len]** represent? What happens if it is not released?

> Hint: pages are allocated for the cache-miss suffix during this request’s Prefill; insert_prefix discovers that another request has already written one of those segments into the tree, so the returned cached_len is greater than old_handle.cached_len.


5. Why must **evict** start from leaves with no references, and who actually returns the pages to the free list?

> Hint: consider only the mini-SGLang implementation; refer to the Radix Cache implementation in mini-SGLang.

## References

- [mini-sglang](https://github.com/sgl-project/mini-sglang)
- [SGLang RadixAttention](https://lmsys.org/blog/2024-01-17-sglang/)
- [SGLang](https://github.com/sgl-project/sglang)

