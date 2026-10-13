# Chapter 4 KV Cache Optimization (Code Walkthrough)

The conceptual chapter explained why KV Cache avoids recomputing historical tokens during Decode. This chapter grounds that result in mini-sglang source code: how a request obtains physical KV locations, how Attention writes newly computed K/V into the cache, and how the next Decode step reads the same cache.

As you read, keep three questions separate: **where the actual K/V floating-point tensors live**, **which physical location corresponds to a token**, and **which historical locations can be reused by later requests**. This chapter focuses on the basic KV Cache read/write path.

## 1 Learning Objectives

After completing this chapter, you should be able to:

1. Locate the implementations of the physical KV Cache, logical-location mapping, and prefix-cache policies in mini-sglang, and explain their distinct responsibilities.
2. Explain the K/V tensor layout of `MHAKVCache`, and how `layer_id`, `out_loc`, and `page_table` jointly determine a write location.
3. Trace `cached_len`, `device_len`, physical-page allocation, KV writes, and historical-KV reads through one Prefill and one Decode step.

## 2 Locating the Source Code

The code covered in this chapter is mainly under mini-sglang's `python/minisgl/` directory:

| File | Purpose |
|---|---|
| `kvcache/base.py` | Defines abstract interfaces for the physical KV pool, prefix cache, cache handles, and capacity accounting |
| `kvcache/__init__.py` | Creates `MHAKVCache` and registers the `naive` and `radix` prefix-cache policies |
| `kvcache/mha_pool.py` | Allocates GPU K/V tensors for every layer and exposes per-layer read/write interfaces |
| `engine/engine.py` | Initializes the KV pool, global `page_table`, and the dummy page used by CUDA Graph |
| `core.py` | Defines `cached_len`, `device_len`, `extend_len`, and batch `out_loc` for requests |
| `scheduler/cache.py` | Manages free pages, fills `page_table`, archives reusable prefixes, and reclaims tail pages |
| `scheduler/prefill.py` | Matches reusable prefixes, locks handles, and prepares request state before Prefill batching |
| `scheduler/scheduler.py` | Allocates physical pages for a batch, calculates `out_loc`, and releases resources when requests finish |
| `attention/fa.py` | Calls `store_kv` to write new K/V before reading the cache for Attention |

The full path can be summarized as:

1. The Scheduler prepares physical locations for the tokens computed in this batch.
2. Attention writes K/V to those locations.
3. Attention reads historical K/V from the same pool.
4. Request state advances, and locations are either retained as reusable cache or returned to the free-page pool.

## 3 Separating Physical KV, Logical Locations, and Prefix Cache

The easiest mistake in KV Cache code is to treat the tensor holding K/V, the mapping from tokens to physical locations, and the cross-request reuse policy as one component. They have different responsibilities:

| Component | Responsibility | Core data |
|---|---|---|
| `MHAKVCache` | Stores the actual K/V floating-point tensors | `_kv_buffer`, per-layer K/V views |
| `page_table` and `Batch.out_loc` | Map logical tokens computed by the batch to physical token slots | `int32` physical-location indices |
| `BasePrefixCache` | Decides whether a historical prefix can be reused across requests | `BaseCacheHandle`, `cached_len`, matched indices |
| `CacheManager` | Coordinates free pages, prefix cache, and the global `page_table` | `free_slots`, page allocation, and reclamation |

`MHAKVCache` does not know which request owns a physical location and does not compare token IDs. It only writes data at the locations given by `out_loc`. The prefix cache does not store FP16 or BF16 K/V vectors either; it records which physical locations correspond to a token sequence. As a result, the actual K/V exists only once, while multiple requests can reference the same locations through different `page_table` rows.

### 3.1 Two Abstract Interfaces

`kvcache/base.py` separates the physical pool from the prefix-reuse policy:

```python
# python/minisgl/kvcache/base.py (excerpt)
class BaseKVCachePool(ABC):
    @abstractmethod
    def k_cache(self, index: int) -> torch.Tensor: ...

    @abstractmethod
    def v_cache(self, index: int) -> torch.Tensor: ...

    @abstractmethod
    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None: ...


@dataclass(frozen=True)
class BaseCacheHandle(ABC):
    cached_len: int

    @abstractmethod
    def get_matched_indices(self) -> torch.Tensor: ...


class BasePrefixCache(ABC):
    @abstractmethod
    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult: ...

    @abstractmethod
    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult: ...

    @abstractmethod
    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None: ...
```

- `BaseKVCachePool` concerns only reading and writing each layer's K/V.
- `BaseCacheHandle` represents one prefix-match result: `cached_len` indicates how many token locations can be reused, and `get_matched_indices()` returns their physical locations.
- `BasePrefixCache` handles matching, insertion, locking, eviction, and capacity accounting. A handle must be locked before the physical locations it returns are used; otherwise, the corresponding prefix can be evicted.

This separation lets an Attention backend operate without knowing whether the active policy is `naive`, which never reuses prefixes, or `radix`, which can reuse prefixes across requests.

### 3.2 `naive` and `radix`: They Decide Reuse, Not K/V Layout

`kvcache/__init__.py` creates `MHAKVCache` as the physical KV pool and selects a prefix policy through a registry:

```python
# python/minisgl/kvcache/__init__.py (excerpt)
def create_kvcache_pool(model_config, num_pages, page_size, dtype, device):
    from .mha_pool import MHAKVCache

    return MHAKVCache(
        num_kv_heads=model_config.num_kv_heads,
        num_pages=num_pages,
        page_size=page_size,
        num_layers=model_config.num_layers,
        head_dim=model_config.head_dim,
        device=device,
        dtype=dtype,
    )


@SUPPORTED_CACHE_MANAGER.register("naive")
def create_naive_cache(device: torch.device):
    return NaivePrefixCache(device=device)


@SUPPORTED_CACHE_MANAGER.register("radix")
def create_radix_cache(device: torch.device):
    return RadixPrefixCache(device=device)
```

`NaivePrefixCache` is the no-prefix-reuse baseline: it always returns `cached_len=0` and an empty physical-location list, and `insert_prefix` does not retain any prefix. It is therefore not an alternative K/V tensor layout; actual K/V remains in the same `MHAKVCache`.

`RadixPrefixCache` records token prefixes and physical locations, so a later matching request can obtain existing locations. Its tree nodes, partial-match splitting, lock reference counts, and LRU leaf eviction are the focus of [Chapter 8](./Chapter8_RadixAttention-and-Prefix-Caching_code.md); this chapter uses only its common interface.

## 4 Initialization: Create the Physical KV Pool and Global Location Table

### 4.1 The Physical Layout of `MHAKVCache`

`MHAKVCache` allocates K/V storage for every layer on the GPU at once:

```python
# python/minisgl/kvcache/mha_pool.py (excerpt)
class MHAKVCache(BaseKVCachePool):
    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        tp_info = get_tp_info()
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        self._kv_buffer = torch.empty(
            (2, num_layers, num_pages, page_size, local_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self._num_layers = num_layers
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)
```

The six dimensions mean:

| Dimension | Meaning |
|---|---|
| `2` | Dimension 0 separates Key from Value |
| `num_layers` | Every Transformer layer owns its own K/V |
| `num_pages` | Number of physical pages |
| `page_size` | Number of token slots per page |
| `local_kv_heads` | Number of KV heads held by the current tensor-parallel rank |
| `head_dim` | Dimension of one attention head |

`local_kv_heads` is not simply global `num_kv_heads`. The code obtains tensor-parallel information through `get_tp_info()` and uses `div_even(..., allow_replicate=True)` to determine the local KV-head count for the current rank. The cache head dimension must therefore match the Attention computation actually running on that rank.

When accessed by layer, `k_cache(layer_id)` and `v_cache(layer_id)` return tensors shaped `[num_pages, page_size, local_kv_heads, head_dim]`:

```python
# python/minisgl/kvcache/mha_pool.py (excerpt)
def k_cache(self, index: int) -> torch.Tensor:
    return self._k_buffer[index]


def v_cache(self, index: int) -> torch.Tensor:
    return self._v_buffer[index]
```

Therefore, **different layers must never share K/V**. Even if two tokens use the same physical slot, `layer_id=0` and `layer_id=1` refer to different layer dimensions.

### 4.2 Physical Pages and Token Slots

`CacheManager` stores only the starting token slot of each free page:

```python
# python/minisgl/scheduler/cache.py (excerpt)
self.free_slots = torch.arange(num_pages, dtype=torch.int32, device=device) * page_size
```

Suppose `page_size=4` and there are three pages. Then `free_slots` is:

```text
[0, 4, 8]
```

They represent the following contiguous physical token-slot ranges:

| Page start | Expanded token slots |
|---:|---|
| 0 | `[0, 1, 2, 3]` |
| 4 | `[4, 5, 6, 7]` |
| 8 | `[8, 9, 10, 11]` |

To write or read a specific token, the system uses the expanded token slot rather than a page number. `_page_to_token` performs this conversion:

```python
# python/minisgl/scheduler/cache.py (excerpt)
def _page_to_token(self, pages: torch.Tensor) -> torch.Tensor:
    if self.page_size == 1:
        return pages
    offsets = torch.arange(self.page_size, device=self.device, dtype=torch.int32)
    return (pages.unsqueeze(1) + offsets).flatten()
```

[Chapter 7](./Chapter7_Paged-KV-Cache-and-Memory-Management.md) examines page allocation, fragmentation, and PagedAttention in depth. The key invariant here is: **`free_slots` must be page-aligned, while `out_loc` for K/V writes is a flattened, token-granular location.**

### 4.3 Engine Creates Both the KV Pool and `page_table`

During initialization, `Engine` determines available pages from free GPU memory, then creates the physical KV pool and the global location table:

```python
# python/minisgl/engine/engine.py (excerpt)
self.num_pages = self._determine_num_pages(init_free_memory, config)
num_tokens = self.num_pages * config.page_size
self.ctx.kv_cache = self.kv_cache = create_kvcache_pool(
    model_config=config.model_config,
    num_pages=self.num_pages + 1,  # +1 for dummy page
    page_size=config.page_size,
    device=self.device,
    dtype=self.dtype,
)

self.max_seq_len = min(config.max_seq_len, num_tokens)
aligned_max_seq_len = _align_up_32(self.max_seq_len)
self.ctx.page_table = self.page_table = torch.zeros(
    (config.max_running_req + 1, aligned_max_seq_len),
    dtype=torch.int32,
    device=self.device,
)
```

Two details are easy to miss:

1. The physical KV pool uses `num_pages + 1`; the additional page is reserved for the CUDA Graph dummy request, while the scheduler manages only the ordinary `num_pages` usable pages.
2. One `page_table` row corresponds to one running request, and its entries store **raw token slots**, not page IDs. The Attention backend converts them to page IDs or keeps them at token granularity according to its own interface.

Thus, `page_table[request_row, position]` means: **which physical token slot holds the K/V for the token at logical position `position` in this request's sequence.**

## 5 Following One Request: From Prefill Writes to Decode Reads

We can now trace a request in actual call order. Its key state is defined by `Req`:

```python
# python/minisgl/core.py (excerpt)
@dataclass(eq=False)
class Req:
    input_ids: torch.Tensor
    table_idx: int
    cached_len: int
    output_len: int
    cache_handle: BaseCacheHandle

    def __post_init__(self) -> None:
        self.device_len = len(self.input_ids)
        self.max_device_len = len(self.input_ids) + self.output_len

    @property
    def extend_len(self) -> int:
        return self.device_len - self.cached_len

    def complete_one(self) -> None:
        self.cached_len = self.device_len
        self.device_len += 1
```

- `cached_len`: length of the prefix with valid K/V that the current Attention operation can reuse.
- `device_len`: sequence length that must be represented on the device before the current forward completes.
- `extend_len = device_len - cached_len`: number of tokens whose K/V this round must compute and write.
- `complete_one()`: after the current token has been processed, makes the former `device_len` the new `cached_len` and reserves one logical position for the next generated token.

### 5.1 Batch Admission: Match a Historical Prefix Without Skipping the Final Prompt Token

For a new request, `PrefillAdder._try_allocate_one` first asks `CacheManager` for a reusable prefix:

```python
# python/minisgl/scheduler/cache.py (excerpt)
def match_req(self, req: PendingReq) -> MatchResult:
    input_len = req.input_len
    assert input_len > 0
    return self.prefix_cache.match_prefix(req.input_ids[: input_len - 1])
```

It deliberately matches only `input_ids[: input_len - 1]` rather than treating the prompt's last token as fully reusable historical context. Prefill must not only build K/V; it must also produce logits at the final position to sample the first output token.

After matching, `PrefillAdder` uses `cached_len` to calculate the remaining extension and locks the handle before using matched physical locations:

```python
# python/minisgl/scheduler/prefill.py (excerpt)
handle = self.cache_manager.match_req(req).cuda_handle
cached_len = handle.cached_len
extend_len = req.input_len - cached_len
estimated_len = extend_len + req.output_len

if estimated_len + self.reserved_size > self.cache_manager.available_size:
    return None
self.cache_manager.lock(handle)
if estimated_len + self.reserved_size > self.cache_manager.available_size:
    return self.cache_manager.unlock(handle)

if cached_len > 0:
    device_ids = self.table_manager.token_pool[table_idx][:cached_len]
    page_entry = self.table_manager.page_table[table_idx][:cached_len]
    device_ids.copy_(req.input_ids[:cached_len].pin_memory(), non_blocking=True)
    page_entry.copy_(handle.get_matched_indices())
```

The handle must stay locked after obtaining `get_matched_indices()`: otherwise another allocation could evict the prefix and make this request's `page_table` refer to physical locations that have already been reassigned.

At this point, only the matched prefix's token IDs and physical locations have been copied. For uncached new tokens, `_add_one_req` first puts token IDs into `token_pool`; new physical pages are allocated together by the Scheduler after the batch has been decided:

```python
# python/minisgl/scheduler/prefill.py (excerpt)
_slice = slice(cached_len, cached_len + chunk_size)
device_ids = self.table_manager.token_pool[table_idx, _slice]
device_ids.copy_(pending_req.input_ids[_slice].pin_memory(), non_blocking=True)
```

### 5.2 Allocate Physical Locations and Produce `out_loc`

Once the batch is assembled, `Scheduler._prepare_batch` allocates physical pages for uncached regions and then constructs the positions and input mapping for this forward:

```python
# python/minisgl/scheduler/scheduler.py (excerpt)
def _prepare_batch(self, batch: Batch) -> ForwardInput:
    self.engine.graph_runner.pad_batch(batch)
    self.cache_manager.allocate_paged(batch.reqs)
    batch.positions = _make_positions(batch, self.device)
    input_mapping = _make_input_tuple(batch, self.device)
    write_mapping = _make_write_tuple(batch, self.device)
    batch.out_loc = self.engine.page_table[input_mapping]
    self.engine.attn_backend.prepare_metadata(batch)
    ...
```

`allocate_paged` allocates physical space only for pages covered by `[cached_len, device_len)`:

```python
# python/minisgl/scheduler/cache.py (excerpt)
def allocate_paged(self, reqs: List[Req]) -> None:
    for req in reqs:
        first_page = div_ceil(req.cached_len, self.page_size)
        last_page = div_ceil(req.device_len, self.page_size)
        if last_page > first_page:
            allocation_info.append((req.table_idx, first_page, last_page))
    if needed_pages > 0:
        allocated = self._page_to_token(self._allocate(needed_pages))
        _write_page_table(self.page_table, allocated, allocation_info, self.page_size)
```

Next, `_make_positions` creates logical positions in `[cached_len, device_len)` for every request, and `_make_input_tuple` pairs each position with that request's `table_idx`. Indexing `page_table` with this pair yields the physical token slots for all newly processed tokens:

```text
batch.positions       = logical positions
input_mapping         = (request-row indices, logical positions)
page_table[input_mapping] = batch.out_loc
```

Therefore, `batch.out_loc` has one entry for every `extend_len` token across the batch, ordered exactly as the model processes the new tokens. It is the sole source of truth for K/V write locations.

### 5.3 Attention: Write New K/V First, Then Read the Complete Context

In the FlashAttention backend, K/V writes and reads happen in the same `forward` method:

```python
# python/minisgl/attention/fa.py (excerpt)
def forward(
    self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
) -> torch.Tensor:
    metadata = batch.attn_metadata
    self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
    return _fa_sgl_impl(
        q=q,
        k_cache=self.kvcache.k_cache(layer_id),
        v_cache=self.kvcache.v_cache(layer_id),
        page_table=metadata.page_table,
        cache_seqlens=metadata.cache_seqlens,
        ...
    )
```

The order matters: the current batch's K/V is written into the physical pool for this layer before that layer's complete K/V cache is passed to the Attention kernel. The current token's Query can therefore access historical K/V as well as the newly written K/V that causal masking permits it to attend to.

`MHAKVCache.store_kv` flattens one layer's `[num_pages, page_size, ...]` storage view across the token dimension, then invokes a kernel for indexed writes:

```python
# python/minisgl/kvcache/mha_pool.py (excerpt)
def store_kv(
    self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
) -> None:
    from minisgl.kernel import store_cache

    store_cache(
        k_cache=self._k_buffer[layer_id].view(self._storage_shape),
        v_cache=self._v_buffer[layer_id].view(self._storage_shape),
        indices=out_loc,
        k=k,
        v=v,
    )
```

The flattened view has shape:

```text
[num_pages * page_size, local_kv_heads, head_dim]
```

If `out_loc=[4, 5]`, K/V for the batch's two new tokens is written to physical token slots 4 and 5. Because `layer_id` is already fixed, these slots affect only the current layer. Each other layer writes the same logical tokens' K/V to its own layer-local storage plane during its own Attention call.

`FlashAttentionBackend.prepare_metadata` also shows that the global `page_table` stores token slots, while the backend takes each page's starting position and converts it to a page ID when required:

```python
# python/minisgl/attention/fa.py (excerpt)
page_table = get_global_ctx().page_table
new_page_table = torch.stack(
    [page_table[req.table_idx, : max_seqlen_k : self.page_size] for req in reqs]
)
if self.page_size > 1:
    new_page_table.div_(self.page_size, rounding_mode="floor")
```

Thus, logical-token-to-physical-token-slot mapping is the system's common internal representation; an Attention backend can convert it to the page-table format required by its kernel at the final step.

### 5.4 Prefill and Decode Write into the Same Pool

Prefill and Decode do not use separate KV Caches. Both use the same path—new tokens obtain `out_loc`, `store_kv` writes them, and Attention reads the cache. Their main difference is `extend_len`:

| Phase | Relationship between `cached_len` and `device_len` | Number of writes this round |
|---|---|---|
| Prefill | Usually contains multiple uncached prompt tokens | `extend_len` can exceed 1 |
| Decode | The token produced in the prior step has become history | Usually `device_len - cached_len = 1` |

After one Decode forward, `Engine.forward_batch` calls `complete_one()` to advance each request's state:

```python
# python/minisgl/engine/engine.py (excerpt)
for req in batch.reqs:
    req.complete_one()
```

This does not copy K/V again. It updates logical boundaries: the token that just participated in forward now has valid K/V, and the next Decode needs to append a location only for a new token. The K/V tensors remain in the same physical `MHAKVCache` pool.

### 5.5 Request Continuation, Completion, and Reusable Prefixes

When Prefill completes and a request continues to Decode, the Scheduler calls `cache_req(req, finished=False)`. When a request finishes normally or is cancelled, `_free_req_resources` calls `cache_req(req, finished=True)`:

```python
# python/minisgl/scheduler/scheduler.py (excerpt)
def _free_req_resources(self, req: Req) -> None:
    self.table_manager.free(req.table_idx)
    self.cache_manager.cache_req(req, finished=True)
```

`cache_req` gives the valid `[0, cached_len)` locations to the prefix-cache policy to decide which portion can be reused, then handles locking and tail reclamation:

```python
# python/minisgl/scheduler/cache.py (excerpt)
def cache_req(self, req: Req, *, finished: bool) -> None:
    insert_ids = req.input_ids[: req.cached_len]
    page_indices = self.page_table[req.table_idx, : req.cached_len]
    old_handle = req.cache_handle
    cached_len, new_handle = self.prefix_cache.insert_prefix(insert_ids, page_indices)
    self.unlock(old_handle)
    self._free(page_indices[old_handle.cached_len : cached_len])
    if finished:
        self._free(page_indices[new_handle.cached_len :])
    else:
        req.cache_handle = new_handle
        self.lock(new_handle)
```

For now, understand these positions in three categories:

1. **Previously matched prefix**: protected by `old_handle` and unlocked once the request is done using it.
2. **Newly computed, whole-page regions successfully registered in the prefix cache**: retained in the physical KV pool so later matching prefixes can reuse them.
3. **A tail that cannot be registered as a reusable prefix, or duplicated locations already owned by another request**: returned to `free_slots` to avoid a leak.

[Chapter 8](./Chapter8_RadixAttention-and-Prefix-Caching_code.md) explains how `insert_prefix` records token-id-to-physical-location mappings in a Radix Tree. [Chapter 7](./Chapter7_Paged-KV-Cache-and-Memory-Management.md) explains how `_allocate`, `_free`, `free_slots`, and page alignment manage GPU memory. The key idea here is: **request completion does not mean all of its K/V is immediately cleared; reusable complete prefixes remain, while non-reusable physical locations are returned.**

## 6 Summary and Exercises

### 6.1 Summary

The KV Cache read/write path in mini-sglang can be summarized as follows:

1. **Physical storage**: `MHAKVCache` uses GPU tensors shaped `[K/V, layer, page, offset, kv_head, head_dim]` to hold every layer's K/V.
2. **Logical mapping**: `page_table` maps a request's logical token positions to physical token slots; the Scheduler builds `batch.out_loc` from locations in `[cached_len, device_len)` for the current round.
3. **Writes and reads**: each Attention layer writes new K/V with `store_kv(k, v, out_loc, layer_id)`, then reads the complete context from its layer-local K/V pool.
4. **Phase advancement**: Prefill can write multiple tokens at once; Decode usually appends one token per step, but both use the same physical pool and write interface.
5. **Reuse and reclamation**: the prefix-cache policy decides which complete prefixes remain for future requests; when a request completes, only non-reusable tail locations return to the free-page list.

### 6.2 Exercises

1. What do dimensions 0, 1, 2, and 3 of `MHAKVCache._kv_buffer` represent? Why can `layer_id` not be omitted?
2. If `page_size=4` and one `free_slots` value is `8`, what physical range does it represent? What does `_page_to_token` expand it into?
3. Why must `Scheduler._prepare_batch` call `allocate_paged` before obtaining `batch.out_loc` through `page_table[input_mapping]`? What happens if the two disagree?

## References

- [mini-sglang](https://github.com/sgl-project/mini-sglang)
- [SGLang](https://github.com/sgl-project/sglang)
