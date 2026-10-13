# 第 4 章 KV Cache 优化（代码走读）

概念章节说明了 KV Cache 为什么能避免在 Decode 阶段反复计算历史 token。本章把这个结论落到 mini-sglang 的源码：一次请求如何获得物理 KV 位置，Attention 如何把新计算出的 K/V 写入缓存，以及下一轮 Decode 如何读取同一份缓存。

阅读时请始终区分三件事：**实际的 K/V 浮点张量存在哪里**、**一个 token 对应哪个物理位置**、**哪些历史位置可以被后续请求复用**。本章重点讲基础 KV Cache 的读写链路；分页分配和显存管理会在[第 7 章](./第7章_PagedKVCache与显存管理.md)展开，Radix 前缀匹配、加锁与淘汰会在[第 8 章](./第8章_RadixAttention与前缀缓存_代码.md)展开。

## 1 本章学习目标

完成本章学习后，你应该能够：

1. 定位 mini-sglang 中物理 KV Cache、逻辑位置映射和前缀缓存策略的实现，并说明它们各自负责什么。
2. 解释 `MHAKVCache` 的 K/V 张量布局，以及 `layer_id`、`out_loc` 和 `page_table` 如何共同确定一次写入的位置。
3. 沿着一次 Prefill 和一次 Decode，追踪 `cached_len`、`device_len`、物理页分配、KV 写入与历史 KV 读取之间的关系。
4. 区分本章的基础读写链路、[第 7 章](./第7章_PagedKVCache与显存管理.md)的分页显存管理，以及[第 8 章](./第8章_RadixAttention与前缀缓存_代码.md)的跨请求前缀复用。

## 2 定位源码

本章参考的代码主要位于 mini-sglang 的 `python/minisgl/` 目录：

| 文件 | 作用 |
|---|---|
| `kvcache/base.py` | 定义物理 KV 池、前缀缓存、缓存句柄和容量统计的抽象接口 |
| `kvcache/__init__.py` | 创建 `MHAKVCache`，并注册 `naive`、`radix` 两种前缀缓存策略 |
| `kvcache/mha_pool.py` | 分配各层 K/V 的 GPU 张量，提供按层读取和写入接口 |
| `engine/engine.py` | 初始化 KV 池、全局 `page_table` 和供 CUDA Graph 使用的 dummy page |
| `core.py` | 定义请求中的 `cached_len`、`device_len`、`extend_len` 和 batch 中的 `out_loc` |
| `scheduler/cache.py` | 管理空闲页、填充 `page_table`、写回可复用前缀和回收尾部页 |
| `scheduler/prefill.py` | 在 Prefill 入批时匹配可复用前缀、锁定句柄并准备请求状态 |
| `scheduler/scheduler.py` | 为 batch 分配物理页、计算 `out_loc`，并在请求结束后释放资源 |
| `attention/fa.py` | 在 Attention 中先调用 `store_kv` 写入新 K/V，再读取缓存做注意力计算 |

可以把整条链路概括为：**Scheduler 为本次计算的 token 准备物理位置 → Attention 把 K/V 写到这些位置 → Attention 从同一池中读取历史 K/V → 请求状态推进，必要时将位置保留为可复用缓存或归还空闲页。**

## 3 正确区分：物理 KV、逻辑位置与前缀缓存

KV Cache 相关代码最容易混淆的地方，是把“存 K/V 的张量”“token 到物理位置的映射”和“跨请求复用策略”当成同一个组件。它们实际上职责不同：

| 组件 | 负责什么 | 核心数据 |
|---|---|---|
| `MHAKVCache` | 保存真正的 K/V 浮点张量 | `_kv_buffer`、每层 K/V view |
| `page_table` 与 `Batch.out_loc` | 将本次计算的逻辑 token 映射到物理 token slot | `int32` 物理位置索引 |
| `BasePrefixCache` | 决定历史前缀能否跨请求复用 | `BaseCacheHandle`、`cached_len`、匹配的 indices |
| `CacheManager` | 在空闲页、前缀缓存和全局 `page_table` 之间协调 | `free_slots`、页分配和回收 |

`MHAKVCache` 不知道某个物理位置属于哪个请求，也不比较 token id；它只根据 `out_loc` 把数据写到指定位置。前缀缓存也不保存 FP16 或 BF16 的 K/V 向量，而是保存“某段 token 已经对应哪些物理位置”。这样，真正的 K/V 仍只有一份，多个请求可以通过不同的 `page_table` 行引用同一批位置。

### 3.1 两套抽象接口

`kvcache/base.py` 把物理池和前缀复用策略分开定义：

```python
# python/minisgl/kvcache/base.py（节选）
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

- `BaseKVCachePool` 只关心每层 K/V 的读取和写入。
- `BaseCacheHandle` 代表一次前缀匹配结果：`cached_len` 表示已有多少 token 的位置可以复用，`get_matched_indices()` 返回这些 token 的物理位置。
- `BasePrefixCache` 负责匹配、插入、加锁、淘汰和容量统计。句柄在返回物理位置后必须先被锁定，否则对应前缀可能被淘汰。

这层接口划分使 Attention 后端不需要知道当前使用的是“完全不复用前缀”的 `naive` 策略，还是能复用跨请求前缀的 `radix` 策略。

### 3.2 `naive` 与 `radix`：决定是否复用，不决定 K/V 布局

`kvcache/__init__.py` 总是创建 `MHAKVCache` 作为物理 KV 池，同时通过注册表选择前缀策略：

```python
# python/minisgl/kvcache/__init__.py（节选）
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

`NaivePrefixCache` 是“不做前缀复用”的基线：它每次返回 `cached_len=0` 和空的物理位置列表，`insert_prefix` 也不会保存任何前缀。因此它并不是另一种 K/V 张量布局，物理 K/V 仍由同一个 `MHAKVCache` 保存。

`RadixPrefixCache` 会把 token 前缀与物理位置登记起来，后续请求命中时就能得到已有位置。它的树节点、部分匹配拆分、锁引用计数和 LRU 叶子淘汰属于第 8 章的主题，本章只使用其统一接口。

## 4 初始化：创建物理 KV 池和全局位置表

### 4.1 `MHAKVCache` 的物理布局

`MHAKVCache` 在 GPU 上一次性分配所有层的 K/V 存储：

```python
# python/minisgl/kvcache/mha_pool.py（节选）
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

这 6 个维度的含义是：

| 维度 | 含义 |
|---|---|
| `2` | 第 0 维区分 Key 和 Value |
| `num_layers` | 每个 Transformer 层各有一份 K/V |
| `num_pages` | 物理页数量 |
| `page_size` | 每页可容纳的 token 数 |
| `local_kv_heads` | 当前 tensor-parallel rank 持有的 KV 头数 |
| `head_dim` | 单个注意力头的维度 |

`local_kv_heads` 不是简单等于全局 `num_kv_heads`。代码通过 `get_tp_info()` 获取 tensor parallel 信息，再用 `div_even(..., allow_replicate=True)` 得到当前 rank 的本地 KV 头数。也就是说，缓存的头维度必须与实际运行该 rank 的 Attention 计算一致。

按层读取时，`k_cache(layer_id)` 与 `v_cache(layer_id)` 返回的形状为 `[num_pages, page_size, local_kv_heads, head_dim]`：

```python
# python/minisgl/kvcache/mha_pool.py（节选）
def k_cache(self, index: int) -> torch.Tensor:
    return self._k_buffer[index]


def v_cache(self, index: int) -> torch.Tensor:
    return self._v_buffer[index]
```

因此，**不同层绝不能共享 K/V**。即便两个 token 的物理 slot 相同，`layer_id=0` 和 `layer_id=1` 也会落在不同的层维度上。

### 4.2 物理页与 token slot

`CacheManager` 的空闲列表只保存每页的起始 token slot：

```python
# python/minisgl/scheduler/cache.py（节选）
self.free_slots = torch.arange(num_pages, dtype=torch.int32, device=device) * page_size
```

假设 `page_size=4`、共有 3 页，那么 `free_slots` 是：

```text
[0, 4, 8]
```

它们分别代表三段连续的物理 token slot：

| page start | 展开后的 token slot |
|---:|---|
| 0 | `[0, 1, 2, 3]` |
| 4 | `[4, 5, 6, 7]` |
| 8 | `[8, 9, 10, 11]` |

需要写入或读取某个具体 token 时，系统使用展开后的 token slot，而不是只使用页号。`_page_to_token` 正是完成这项转换：

```python
# python/minisgl/scheduler/cache.py（节选）
def _page_to_token(self, pages: torch.Tensor) -> torch.Tensor:
    if self.page_size == 1:
        return pages
    offsets = torch.arange(self.page_size, device=self.device, dtype=torch.int32)
    return (pages.unsqueeze(1) + offsets).flatten()
```

后续章节会深入解释页分配、碎片和 PagedAttention；此处需要记住的关键不变量是：**`free_slots` 必须页对齐，而写入 K/V 的 `out_loc` 是 token 粒度的扁平位置。**

### 4.3 Engine 同时创建 KV 池与 `page_table`

`Engine` 初始化时先根据可用显存决定可用页数，再创建物理 KV 池和全局位置表：

```python
# python/minisgl/engine/engine.py（节选）
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

这里有两个容易忽略的细节：

1. 物理 KV 池使用 `num_pages + 1`，多出的 1 页专门供 CUDA Graph 的 dummy request 使用；调度器管理的可用页仍是普通的 `num_pages`。
2. `page_table` 的一行对应一个运行中的请求，表项保存的是**原始 token slot**，不是 page id。Attention 后端会根据自己的接口要求把它转换成页号或保留为 token 粒度。

因此，`page_table[request_row, position]` 的语义是：**这个请求逻辑序列中 position 位置的 token，它的 K/V 应写入或读取哪个物理 token slot。**

## 5 沿一次请求走读：从 Prefill 写入到 Decode 读取

现在沿真实调用顺序追踪一次请求。核心状态来自 `Req`：

```python
# python/minisgl/core.py（节选）
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

- `cached_len`：已经有有效 KV、可供当前 Attention 复用的前缀长度。
- `device_len`：当前一轮 forward 结束前，需在设备上表示的序列长度。
- `extend_len = device_len - cached_len`：这一轮真正要计算并写入 K/V 的 token 数。
- `complete_one()`：本轮 token 已经完成计算后，把原 `device_len` 变成新的 `cached_len`，再为下一个待生成 token 预留一个逻辑位置。

### 5.1 入批：匹配已有前缀，但不跳过最后一个 prompt token

对新请求，`PrefillAdder._try_allocate_one` 首先向 `CacheManager` 查询可复用前缀：

```python
# python/minisgl/scheduler/cache.py（节选）
def match_req(self, req: PendingReq) -> MatchResult:
    input_len = req.input_len
    assert input_len > 0
    return self.prefix_cache.match_prefix(req.input_ids[: input_len - 1])
```

它故意只匹配 `input_ids[: input_len - 1]`，不把 prompt 的最后一个 token 当作“完全无需计算”的历史前缀。原因是 Prefill 不仅要建立 K/V，还需要通过最后一个位置产生 logits，进而采样第一个输出 token。

匹配后，`PrefillAdder` 使用 `cached_len` 计算本轮仍需扩展的长度，并在真正使用命中的物理位置前锁定句柄：

```python
# python/minisgl/scheduler/prefill.py（节选）
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

句柄必须在取到 `get_matched_indices()` 后保持锁定：否则另一个分配动作可能淘汰这段前缀，使当前请求的 `page_table` 指向已经被重新分配的物理位置。

此时只复制了命中前缀的 token id 和物理位置。对于未命中的新 token，`_add_one_req` 先把 token id 放进 `token_pool`；新的物理页会在 batch 确定后由 Scheduler 统一分配：

```python
# python/minisgl/scheduler/prefill.py（节选）
_slice = slice(cached_len, cached_len + chunk_size)
device_ids = self.table_manager.token_pool[table_idx, _slice]
device_ids.copy_(pending_req.input_ids[_slice].pin_memory(), non_blocking=True)
```

### 5.2 分配物理位置并生成 `out_loc`

Batch 组装完成后，`Scheduler._prepare_batch` 先分配未命中部分所需的物理页，再构建这轮 forward 的位置和输入映射：

```python
# python/minisgl/scheduler/scheduler.py（节选）
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

`allocate_paged` 只为 `[cached_len, device_len)` 覆盖到的页分配空间：

```python
# python/minisgl/scheduler/cache.py（节选）
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

随后 `_make_positions` 为每个请求构造 `[cached_len, device_len)` 的逻辑位置，`_make_input_tuple` 将这些位置与该请求的 `table_idx` 配对。以这个二维索引访问 `page_table` 后，就得到本轮所有新增 token 的物理 token slot：

```text
batch.positions       = 逻辑位置
input_mapping         = (请求行号, 逻辑位置)
page_table[input_mapping] = batch.out_loc
```

因此，`batch.out_loc` 的长度等于 batch 中所有请求的 `extend_len` 之和，并按模型实际处理新增 token 的顺序排列。它是 K/V 写入的唯一位置依据。

### 5.3 Attention：先写新 K/V，再读取完整上下文

FlashAttention 后端中，K/V 写入和读取发生在同一个 `forward` 中：

```python
# python/minisgl/attention/fa.py（节选）
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

顺序很重要：当前 batch 的 K/V 先写进对应层的物理池，再把该层的完整 K/V 缓存传给 Attention kernel。这样当前 token 的 Query 既能访问已有历史，也能访问本轮新写入、在因果掩码允许范围内的 K/V。

`MHAKVCache.store_kv` 将某层的 `[num_pages, page_size, ...]` 存储视图压平成连续 token 维度，再调用 kernel 做按索引写入：

```python
# python/minisgl/kvcache/mha_pool.py（节选）
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

这里的扁平视图形状为：

```text
[num_pages * page_size, local_kv_heads, head_dim]
```

假设 `out_loc=[4, 5]`，那么本轮两个新增 token 的 K/V 就分别写到第 4、5 个物理 token slot。由于 `layer_id` 已经固定，这两个 slot 只影响当前层；其他层会在各自的 Attention 调用中写入相同逻辑 token 对应的本层 K/V。

`FlashAttentionBackend.prepare_metadata` 也说明全局 `page_table` 存的是 token slot，而后端会按自己的要求取每页的起始位置并转成 page id：

```python
# python/minisgl/attention/fa.py（节选）
page_table = get_global_ctx().page_table
new_page_table = torch.stack(
    [page_table[req.table_idx, : max_seqlen_k : self.page_size] for req in reqs]
)
if self.page_size > 1:
    new_page_table.div_(self.page_size, rounding_mode="floor")
```

因此，逻辑 token → 物理 token slot 的映射是系统内部统一表示；不同 Attention 后端可以在最后一步转换成各自 kernel 所需的 page table 格式。

### 5.4 Prefill 与 Decode 写入同一份池

Prefill 和 Decode 并不使用两套 KV Cache。二者都遵循相同的“本轮新增 token 生成 `out_loc` → `store_kv` 写入 → Attention 读取缓存”路径，差别主要在 `extend_len`：

| 阶段 | `cached_len` 与 `device_len` 的关系 | 本轮写入数量 |
|---|---|---|
| Prefill | 通常有多个未缓存 prompt token | `extend_len` 可以大于 1 |
| Decode | 上一步产生的 token 已成为历史 | 通常 `device_len - cached_len = 1` |

一次 Decode forward 完成后，`Engine.forward_batch` 调用 `complete_one()` 推进每个请求的状态：

```python
# python/minisgl/engine/engine.py（节选）
for req in batch.reqs:
    req.complete_one()
```

其含义不是重新复制 K/V，而是更新请求的逻辑边界：刚刚参与 forward 的 token 现在已经有有效 K/V，下一次 Decode 只需为新 token 再追加一个位置。KV 张量本体一直留在 `MHAKVCache` 的相同物理池中。

### 5.5 请求继续、结束与可复用前缀

Prefill 完成且请求仍需 Decode 时，Scheduler 调用 `cache_req(req, finished=False)`；请求正常结束或取消时，经 `_free_req_resources` 调用 `cache_req(req, finished=True)`：

```python
# python/minisgl/scheduler/scheduler.py（节选）
def _free_req_resources(self, req: Req) -> None:
    self.table_manager.free(req.table_idx)
    self.cache_manager.cache_req(req, finished=True)
```

`cache_req` 将已经有效的 `[0, cached_len)` 位置交给前缀缓存策略决定哪些部分可复用，并处理锁和尾部回收：

```python
# python/minisgl/scheduler/cache.py（节选）
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

这一段的详细语义可以暂时按三类位置理解：

1. **原本已命中的前缀**：由 `old_handle` 保护，当前请求使用完后需要解锁。
2. **本请求新计算、成功登记为前缀缓存的整页部分**：保留在物理 KV 池中，后续相同前缀可以复用。
3. **不能登记为可复用前缀的尾部或已被其他请求占用的重复部分**：需要归还给 `free_slots`，避免泄漏。

## 6 总结与测试题

### 6.1 课程总结

mini-sglang 的 KV Cache 读写主线可以概括为：

1. **物理存储**：`MHAKVCache` 用 `[K/V, layer, page, offset, kv_head, head_dim]` 的 GPU 张量保存所有层的 K/V。
2. **逻辑映射**：`page_table` 将请求中的逻辑 token 位置映射到物理 token slot；Scheduler 从本轮 `[cached_len, device_len)` 的映射构造 `batch.out_loc`。
3. **写入与读取**：每层 Attention 先通过 `store_kv(k, v, out_loc, layer_id)` 写入新增 K/V，再从同层 K/V 池读取完整上下文完成注意力计算。
4. **阶段推进**：Prefill 可批量写入多个 token；Decode 通常每轮只追加一个 token，但两者都使用同一套物理池和写入接口。
5. **复用与回收**：前缀缓存策略决定哪些完整前缀保留给未来请求；请求结束后，不能复用的尾部位置才会回到空闲页列表。

### 6.2 测试题

1. `MHAKVCache._kv_buffer` 的第 0、1、2、3 个维度分别表达什么？为什么 `layer_id` 不能省略？
2. 若 `page_size=4`，`free_slots` 中一个值为 `8`，它代表什么物理范围？`_page_to_token` 会将它展开为什么？
3. `Scheduler._prepare_batch` 为什么要先调用 `allocate_paged`，再通过 `page_table[input_mapping]` 得到 `batch.out_loc`？若两者不一致，`store_kv` 会产生什么问题？

## 参考资料

- [mini-sglang](https://github.com/sgl-project/mini-sglang)
- [SGLang](https://github.com/sgl-project/sglang)
