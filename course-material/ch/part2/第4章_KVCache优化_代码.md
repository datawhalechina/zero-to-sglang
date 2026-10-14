# 第 4 章 KV Cache 优化（代码走读）

概念章节说明了 KV Cache 为什么能避免在 Decode 阶段反复计算历史 token。本章把这个结论落到 mini-sglang 的源码：一次请求如何获得物理 KV 位置，Attention 如何把新计算出的 K/V 写入缓存，以及下一轮 Decode 如何读取同一份缓存。

阅读时请始终区分三件事：**实际的 K/V 浮点张量存在哪里**、**一个 token 对应哪个物理位置**、**哪些历史位置可以被后续请求复用**。本章重点讲基础 KV Cache 的读写链路。

## 1 本章学习目标

完成本章学习后，你应该能够：

1. 定位 mini-sglang 中物理 KV Cache、逻辑位置映射和前缀缓存策略的实现，并说明它们各自负责什么。
2. 解释 `MHAKVCache` 的 K/V 张量布局，以及 `layer_id`、`out_loc` 和 `page_table` 如何共同确定一次写入的位置。
3. 沿着一次 Prefill 和一次 Decode，追踪 `cached_len`、`device_len`、物理页分配、KV 写入与历史 KV 读取之间的关系。

## 2 定位源码

本章参考的代码主要位于 mini-sglang 的 `python/minisgl/` 目录。按职责可分为四组：

**1. 接口与入口**

| 文件 | 作用 |
|---|---|
| `kvcache/base.py` | 定义物理 KV 池的读写抽象接口 |
| `kvcache/__init__.py` | 创建 KV 池，并注册缓存策略 |

**2. 存储与初始化**

| 文件 | 作用 |
|---|---|
| `kvcache/mha_pool.py` | 在 GPU 上存放各层 K/V 张量，提供按层读取和写入接口 |
| `engine/engine.py` | 启动时初始化 KV 池、全局 `page_table` 等运行时对象 |

**3. 状态与调度**

| 文件 | 作用 |
|---|---|
| `core.py` | 记录请求进度和本轮需要写入 KV 的逻辑范围 |
| `scheduler/cache.py` | 分配与回收页，管理前缀缓存 |
| `scheduler/prefill.py` | Prefill 时匹配并锁定可复用的前缀 |
| `scheduler/scheduler.py` | 组织 batch，确定本轮 K/V 写入位置，清理结束的请求 |

**4. 计算**

| 文件 | 作用 |
|---|---|
| `attention/fa.py` | 在 Attention 前向过程中读写 K/V Cache 并完成注意力计算 |

可以把整条链路概括为：

1. Scheduler 为本次计算的 token 准备物理位置。
2. Attention 在前向过程中对 KV Cache 进行相应读写。
3. 请求状态推进，必要时将位置保留为可复用缓存或归还空闲页。

## 3 正确区分：物理 KV、逻辑位置与前缀缓存

KV Cache 相关代码最容易混淆的地方，是把存 K/V 的张量、token 到物理位置的映射和跨请求复用策略当成同一个组件。它们实际上职责不同：

| 组件 | 负责什么 | 核心数据 |
|---|---|---|
| `MHAKVCache` | 保存真正的 K/V 浮点张量 | `_kv_buffer`、每层 K/V view |
| `page_table` 与 `Batch.out_loc` | 将本次计算的逻辑 token 映射到物理 token slot | `int32` 物理位置索引 |
| `BasePrefixCache` | 决定历史前缀能否跨请求复用 | `BaseCacheHandle`、`cached_len`、匹配的 indices |
| `CacheManager` | 在空闲页、前缀缓存和全局 `page_table` 之间协调 | `free_slots`、页分配和回收 |

`MHAKVCache` 不知道某个物理位置属于哪个请求，也不比较 token id；它只根据 `out_loc` 把数据写到指定位置。前缀缓存也不保存 FP16 或 BF16 的 K/V 向量，而是保存某段 token 已经对应哪些物理位置。这样，真正的 K/V 仍只有一份，多个请求可以通过不同的 `page_table` 行引用同一批位置。

### 3.1 `BaseKVCachePool`：物理 KV 池的读写接口

`kvcache/base.py` 中的 `BaseKVCachePool` 定义了物理 KV 池对 Attention 后端提供的最小接口：

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

    @property
    @abstractmethod
    def device(self) -> torch.device: ...

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype: ...

    @property
    @abstractmethod
    def num_layers(self) -> int: ...
```

前三个方法描述读写操作：

- `k_cache(layer_id)`：取得指定层的完整 K 缓存视图。
- `v_cache(layer_id)`：取得指定层的完整 V 缓存视图。
- `store_kv(k, v, out_loc, layer_id)`：把本轮新计算出的 K/V 写到 Scheduler 已确定的物理 token slot。

后三个属性提供池的运行时元数据：所在的 `device`、K/V 的 `dtype` 和 `num_layers`。本章后续重点仍放在前三个读写操作。

物理 KV 池不理解请求归属、不比较 token ID，也不决定哪些位置能够复用。它只按 `layer_id` 选择层，按 `out_loc` 选择写入位置；具体请求使用哪些位置由 `page_table` 决定。

### 3.2 `naive` 与 `radix`：决定是否复用，不改变 KV 池

`kvcache/__init__.py` 创建 `MHAKVCache` 作为物理 KV 池，同时注册不同的前缀复用策略：

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

`naive` 和 `radix` 不是两种 KV 张量布局：它们只决定已经写入物理 KV 池的位置能否被后续请求复用。无论采用哪种策略，Attention 的读写对象都是同一个 `MHAKVCache`，调用接口也是 `k_cache`、`v_cache` 和 `store_kv`。Radix 前缀匹配、锁定与淘汰的具体实现属于[第 8 章](./第8章_RadixAttention与前缀缓存_代码.md)。

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

代码通过 `k_cache(layer_id)` 和 `v_cache(layer_id)` 读取指定层的 K/V：

```python
# python/minisgl/kvcache/mha_pool.py（节选）
def k_cache(self, index: int) -> torch.Tensor:
    return self._k_buffer[index]


def v_cache(self, index: int) -> torch.Tensor:
    return self._v_buffer[index]
```

以读取 K 为例，数据沿着以下层次取出：

```text
全部 K/V：_kv_buffer
形状：[2, num_layers, num_pages, page_size, local_kv_heads, head_dim]
    ↓ 初始化时取第 0 项，得到 K 的视图
所有层的 K：_k_buffer = _kv_buffer[0]
形状：[num_layers, num_pages, page_size, local_kv_heads, head_dim]
    ↓ 读取时按 layer_id 选择层
指定层的 K：_k_buffer[layer_id]
形状：[num_pages, page_size, local_kv_heads, head_dim]
```

读取 V 的过程相同，只是从 `_kv_buffer[1]` 得到 `_v_buffer`。`_k_buffer` 和 `_v_buffer` 都是对 `_kv_buffer` 的视图，并不会复制数据；`k_cache(layer_id)` 和 `v_cache(layer_id)` 最终拿到的是该层所有物理页的 K 或 V 张量。

最终四个维度依次表示：页号、页内 token 位置、本地 KV 头、头内维度。具体请求应读取其中哪些位置，再由 `page_table` 和 Attention 的有效长度 metadata 确定。因此，**不同层绝不能共享 K/V**：即便两个 token 的物理 slot 相同，`layer_id=0` 和 `layer_id=1` 也会落在不同的层维度上。

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

它故意只匹配 `input_ids[: input_len - 1]`，不把 prompt 的最后一个 token 当作完全无需计算的历史前缀。原因是 Prefill 不仅要建立 K/V，还需要通过最后一个位置产生 logits，进而采样第一个输出 token。

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

一批请求开始计算前，要先确定新算出的 K/V 存到哪里。Scheduler 会分配需要的新缓存页，再把本轮每个 token 对应的存放位置记在 `out_loc` 中；后面算出 K/V 后，Attention 就按这个列表对缓存进行写入。

`Scheduler._prepare_batch` 在分配后构建这轮 forward 的位置和输入映射：

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

这样的顺序很重要：当前 batch 的 K/V 先写进对应层的物理池，再把该层的完整 K/V 缓存传给 Attention kernel。这样当前 token 的 Query 既能访问已有历史，也能访问本轮新写入、在因果掩码允许范围内的 K/V。

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

因此，逻辑 token 到 物理 token slot 的映射是系统内部统一表示；不同 Attention 后端可以在最后一步转换成各自 kernel 所需的 page table 格式。

### 5.4 请求继续、结束与可复用前缀

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
