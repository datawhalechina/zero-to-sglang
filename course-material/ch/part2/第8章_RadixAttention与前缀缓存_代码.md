# 第 8 章 RadixAttention 与前缀缓存（代码）

前一章已经介绍了 Radix Tree、最长公共前缀、KV Cache Pool，以及 $match \to lock \to allocate \to insert \to unlock \to evict/free$ 这条缓存生命周期。本章进一步把这些抽象概念落到 mini-SGLang 的源码中：我们跟随一条带有缓存命中的请求，从前缀匹配开始，依次观察索引如何生成、物理页如何分配、KV 如何写入，以及请求结束时资源如何回收。

阅读时可以把“Radix Tree 负责记录映射，KV Cache Pool 负责保存数据”作为主线。建议先阅读本章的[概念文档](./第8章_RadixAttention与前缀缓存.md)，再对照本文阅读。

## 1 本章学习目标

完成本章学习后，应该能够清楚回答以下问题：

- 定位 mini-sglang 中 Prefix Cache、page table 与 KV Cache Pool 的实现，并说明三者如何协作？
- 一次请求从匹配前缀到处理结束的完整生命周期，在代码中是如何串联实现的？
- Prefix Cache 的核心操作各自做了什么？


## 2 定位源码

本章参考的代码文件都位于 **python/minisgl/**：

| 文件 | 作用 |
|------|------|
| kvcache/base.py | 定义 Prefix Cache 与 KV Cache Pool 的抽象接口 |
| kvcache/radix_cache.py | Radix Tree、缓存句柄、匹配、插入、加锁与可淘汰叶子上的淘汰逻辑 |
| scheduler/cache.py | 连接 Radix Tree、free_slots 与 page table|
| scheduler/prefill.py | 入批前匹配前缀、锁定句柄、写入命中段；chunked prefill |
| scheduler/table.py | 请求槽位与 token_pool / page table 行的分配与释放 |
| scheduler/decode.py | Decode 运行队列与 inflight_tokens 预估 |
| scheduler/scheduler.py | 组 batch、分配新页、准备 Attention 输入；结束/取消时回收资源 |
| kvcache/mha_pool.py | 在 GPU 上按层保存 K、V 张量 |
| kernel/csrc/jit/store.cu | KV 缓冲区写入的 CUDA 实现 |
| kernel/csrc/src/radix.cpp | 加速比较 Radix Tree 中的 key |
| kernel/radix.py | 上述 radix C++ 的 Python 封装 |

下面先定位这些文件各自承担的职责，再沿请求生命周期把它们串起来。阅读后续代码时，遇到一个函数可以先问：它是在改变“映射”，还是在操作“物理 KV”？

## 3 正确区分：映射表 vs 真正的 KV 存储

在 RadixAttention 相关代码里，最容易混淆的是**映射关系**和**真正的 KV Cache Pool**。token id 序列、物理槽位等在前面概念章节已讲过，这里不再重复，只区分两个组件：

| 组件 | 职责 | 存什么 |
|------|------|--------|
| **Radix Tree** | 前缀匹配与索引 | **token id 序列 → 物理槽位 indices** 的映射（以及树拓扑、ref_count 等） |
| **MHAKVCache** | 物理 KV 存储池 | 连续的 K/V Tensor 缓冲区（paged layout） |

Radix Tree **不存任何 KV 浮点数据**，只记录：某段 token 序列对应的 KV 落在物理缓冲区的哪些槽位。

真正的 KV Tensor 由 **MHAKVCache** 分配对应空间，layers/attention.py 算完后通过 CUDA kernel（**store.cu → store_cache**）按 indices 写入。

**1. 物理存储：MHAKVCache**

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
            indices=out_loc,  # 事先分配好的物理槽位索引
            k=k, v=v,
        )
```

- **_kv_buffer** 是 **KV 本体** 的主要显存占用（paged layout）。
- **out_loc**（indices）来自调度阶段（page table 等），告诉 kernel 把当前算出的 KV 写到哪些物理槽位。


**2. 映射表：RadixTreeNode**

```python
# python/minisgl/kvcache/radix_cache.py
class RadixTreeNode:
    def set_key_value(self, key: torch.Tensor, value: torch.Tensor):
        self._key = key
        self._value = value
        self._length = len(key)

class RadixCacheHandle:
    def get_matched_indices(self) -> torch.Tensor:
        # 从当前节点沿 parent 走到 root，拼接沿途 node.value
        # 得到完整物理槽位列表，交给 attention 使用
        ...
```

- 每个节点只保存一段连续 token 的 **indices**，不保存 float16/bfloat16 的 K/V。
- **match_prefix**：按 token 序列在树上匹配最长前缀；必要时会 **split_at** 调整拓扑，返回 handle（含 **prefix_len** 与节点）。
- **insert_prefix**：在计算与 **store_kv** 完成之后，把 **(token_ids, page_indices)** 登记进树，供后续请求复用。
- 真正的 KV 存储始终只在 **MHAKVCache** 里，而 Radix Tree 只负责这段 token id 到这些物理槽位的映射记录。

***注**：命中缓存时只需用 indices 让 attention 读已有数据；未命中缓存部分才计算并 **store_kv**，再在适当时机调用 **insert_prefix** 。*

一句话概括 Radix Tree 与 MHAKVCache 的核心就是——**Radix Tree = 索引（token id → 物理槽位）；MHAKVCache = 真正存放 KV 的显存池**。

理解了索引和物理池的关系之后，后面的代码就可以按一个问题来阅读：这段 token 的索引从哪里来，什么时候受到保护，最后又由谁释放？接下来沿着一次 cache hit 请求的完整路径，串起各组件的协作关系。

## 4 沿一次缓存命中走读源码

按照调用顺序展开：创建缓存对象，再用 match_prefix 找到并锁定命中路径，为未命中部分预留和分配页，在 forward 中由 store_kv 写入物理 KV，最后用 insert_prefix 把可复用的 indices 以及相关的 token id 序列登记回 Radix Tree。

---

### 4.1 压缩路径节点：RadixTreeNode

缓存索引本身如何组织，在 mini-sglang 提供 naive 与 radix 两种前缀缓存策略：naive 不做跨请求前缀复用；radix 则用压缩路径的 Radix Tree，按 token id 前缀共享物理页索引。后面的匹配、加锁和淘汰都建立在这棵树上：

```python
# kvcache/__init__.py（节选）
@SUPPORTED_CACHE_MANAGER.register("radix")
def create_radix_cache(device: torch.device):
    return RadixPrefixCache(device=device)

def create_prefix_cache(device: torch.device, type: str) -> BasePrefixCache:
    return SUPPORTED_CACHE_MANAGER[type](device)
```

这里选择 **radix** ，由 Scheduler 初始化 **CacheManager** 为后续请求处理提供基础。它同时持有：**Prefix Cache**（RadixPrefixCache）、**空闲页列表**（free_slots）和**全局 page table**（请求处理时使用的槽位映射）。


Radix Tree 的一个节点表示**一段连续 token 序列**（可跨多页），而不是单个 token：

```python
# kvcache/radix_cache.py（节选）
class RadixTreeNode:
    def __init__(self, key_fn, tic=None):
        self.children = {}
        self._parent = None
        self.ref_count = 0
        self.timestamp = tic or time.monotonic_ns()

    def set_key_value(self, key, value):
        assert len(key) == len(value)
        self._key = key      # token id 段
        self._value = value  # 对应物理槽位 indices
        self._length = len(key)
```

当匹配停在某节点中间时，要在分歧处拆出公共前缀节点（ _tree_walk 里触发）：

```python
def split_at(self, pos):
    parent = self.parent
    new_node = RadixTreeNode(self.key_fn, self.timestamp)
    new_node.set_key_value(self._key[:pos], self._value[:pos])
    new_node.set_parent(parent)
    new_node.ref_count = self.ref_count   # 继承引用计数

    self.set_key_value(self._key[pos:], self._value[pos:])
    self.set_parent(new_node)
    return new_node
```

例如原节点 key 为 [10,11,12,13]，本次只匹配到 [10,11]。split_at(2) 后，新节点持有公共前缀 [10,11]，原节点缩成后缀 [12,13] 并挂到新节点下。物理 KV 不搬移，只拆分 indices；公共段仍被原引用保护，因此新节点继承 ref_count。

到这里，我们知道了树节点如何保存和拆分索引。接下来要回答的是：面对一条新的 input_ids，系统怎样沿这棵树找到最长可复用前缀？

---

### 4.2 最长前缀匹配：从 input_ids 得到可复用范围

树初始化完成后，处理请求的第一步就是匹配：在 Radix Tree 上找出可复用的最长前缀，并得到一个 **handle**。这个 handle 同时携带 `cached_len` 和匹配节点，后续的 page table 填充与 lock 都依赖它。

子节点字典的 key 由 `page_size` 决定：`page_size=1` 时使用下一个 token id；`page_size>1` 时使用对应完整 token id 的元组：

```python
def _get_key_fn(page_size):
    if page_size == 1:
        return lambda x: x[0].item()
    return lambda x: tuple(x[:page_size].tolist())
```

最长前缀匹配在 _tree_walk 中完成：

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

        if match_len != node.length:  # 停在节点中间 → 拆分
            node = node.split_at(match_len)
            node.timestamp = tic
            return node, prefix_len

        node.timestamp = tic  # 整节点命中，刷新 LRU 时间

    return node, prefix_len
```

关键点：

- **get_match_len**：用 fast_compare_key 比较节点 _key 与剩余 input_ids，返回本段可匹配长度。
- 匹配长度按 **page_size 向下对齐**，不共享非整页前缀。其中命中路径上的节点更新 **timestamp**，淘汰时优先保留更近使用的叶子前缀。
- **仅当匹配停在节点中部**（match_len != node.length）时调用 **split_at**，再返回。

match_prefix 把结果封进 handle：

```python
def match_prefix(self, input_ids):
    node, prefix_len = self._tree_walk(input_ids)
    return MatchResult(RadixCacheHandle(prefix_len, node))  # cached_len = prefix_len
```

需要把命中段写进该请求的 page table 时，再调 **handle.get_matched_indices()** ：从当前节点沿 parent 走到 root，收集各段 value，反转后 torch.cat，得到按序列顺序的物理索引。

***说明**：即使树上没有任何可复用前缀，仍会走 match_prefix，此时 `prefix_len == 0`。这并不意味着请求被跳过；未命中部分会在后续分配页并完成 forward、store_kv 后，再由 insert_prefix 登记到 Radix Tree 中*。

匹配只给出了“哪些内容可以复用”，还没有真正为本次请求准备可写空间。下一步因此从只读的匹配结果进入调度阶段： lock 命中前缀，并为未命中后缀预留资源。

---

### 4.3 从匹配结果到可执行 Batch：分配 page 并加锁

匹配结束后，Scheduler 中已经有了 **handle**。此时命中前缀仍只是一个可复用范围，必须先锁定它，避免在本轮调度期间被淘汰；同时还要估算未命中后缀和后续 Decode 所需的空间。真正的物理 page 会等 Batch 确定、进入 forward 前再分配。

mini-sglang 处理 Prefill 阶段的计算时采用先到先处理策略，即按 **pending_list** 顺序尝试加入请求，一旦空间不足就 `break`，以保证 kernel 计算的稳定性：

```python
# scheduler/prefill.py（节选）
def try_add_one(self, pending_req: PendingReq) -> Req | None:
    if self.token_budget <= 0:
        return None
    # 已有 chunked 进度：直接继续切下一块
    if chunked_req := pending_req.chunked_req:
        return self._add_one_req(...)
    # 新请求：先试分配，再正式加入
    if resource := self._try_allocate_one(pending_req):
        ...
    return None

def schedule_next_batch(self, prefill_budget: int) -> Batch | None:
    if len(self.pending_list) == 0:
        return None
    adder = PrefillAdder(
        token_budget=prefill_budget,
        reserved_size=self.decode_manager.inflight_tokens,  #  decode 占用的预估空间
        ...
    )
    reqs: List[Req] = []
    chunked_list: List[PendingReq] = []
    for pending_req in self.pending_list:
        if req := adder.try_add_one(pending_req):
            pending_req.chunked_req = None
            if isinstance(req, ChunkedReq):
                # 半成品 Req
                ...
            reqs.append(req) # 普通 Req
        else:
            break  # 空间不足，停止继续添加
    if len(reqs) == 0:
        return None
    # 未完成的 chunked 请求放回队头，避免饥饿
    self.pending_list = chunked_list + self.pending_list[len(reqs):]
    return Batch(reqs=reqs, phase="prefill")
```

---

对新请求，会先走 **PrefillAdder._try_allocate_one** 做空间预估与 lock，为本次 Prefill 扩展以及后续 Decode 预留空间：

```python
# scheduler/prefill.py（节选）
def _try_allocate_one(self, req: PendingReq) -> Tuple[BaseCacheHandle, int] | None:
    if self.table_manager.available_size == 0:
        return None

    # match_req 匹配 input_ids[:input_len-1]（故意少匹配最后一个 token，留给后续 extend）
    handle = self.cache_manager.match_req(req).cuda_handle
    cached_len = handle.cached_len
    extend_len = req.input_len - cached_len
    estimated_len = extend_len + req.output_len

    # 第一次检测：Prefill 扩展 + Decode 预留是否足够
    if estimated_len + self.reserved_size > self.cache_manager.available_size:
        return None
    self.cache_manager.lock(handle)

    # 第二次检测：命中前缀加锁后，有没有让可使用的预分配空间不足
    if estimated_len + self.reserved_size > self.cache_manager.available_size:
        return self.cache_manager.unlock(handle)  # unlock 返回 None

    table_idx = self.table_manager.allocate()
    if cached_len > 0:
        # 同时复制命中部分的 token id 与 page table 物理索引
        ...
    return handle, table_idx
```

**上面的两次检查分别发生在加锁前后：第一次判断当前资源是否足够，第二次确认加锁带来的状态变化没有使预算分配空间失效**。理解这一点后，再看 `available_size` 的含义会更清楚：它表示当前已淘汰的前缀缓存和空闲页对应的 token 数量：

```python
@property
def available_size(self) -> int:
    return self.prefix_cache.size_info.evictable_size + len(self.free_slots) * self.page_size
```

预分配空间成功后，通过 **PrefillAdder._add_one_req** 正式提交本轮要计算的部分。此时只写入 token id，并按 budget 做 chunk，**此时物理 page 尚未分配**。

```python
# scheduler/prefill.py（节选）
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

    # 只更新 token ids；新 page 的物理分配在 scheduler 调用 allocate_paged 时进行
    _slice = slice(cached_len, cached_len + chunk_size)
    device_ids = self.table_manager.token_pool[table_idx, _slice]
    device_ids.copy_(...)
    return CLS(...)
```

为避免超长请求一次 Prefill 导致峰值显存过高（OOM），会对请求做 **chunk 分块**——每一轮只处理该请求的一个分块也可与其他较短请求组成同一 Batch，而不是一轮中把同一请求的多个分块全部算完。

若本轮仍是 `ChunkedReq == True`，会在 **schedule_next_batch()** 里被放回 **pending_list** 队头，优先在下一轮继续处理，避免出现“饥饿”现象。此时我们只有请求的逻辑范围和资源预算，还没有把新页写入 page table。

---

当 Batch 组装完成、即将真正执行 Prefill forward 时，调度器才把预算兑现为物理映射：调用 **allocate_paged**，并且**只为未命中缓存的区间分配新的物理 page**：

```python
# scheduler/cache.py（节选）
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

[0, cached_len)的 page table 已在命中时由 **handle.get_matched_indices()** 填好。这里从 ceil(cached_len / page_size) 起，只为 [cached_len, device_len) 涉及的新页分配空间。其中 **_page_to_token** 将 page 起始 slot 展开为完整 token 索引再写入 page table。

若 free list 不够，**_allocate** 先按需从 Radix Tree 淘汰子叶节点得到 token slot，把每页起始 slot 放回 **free_slots**，再完成分配：

```python
# scheduler/cache.py（节选）
def _allocate(self, needed_pages: int) -> torch.Tensor:
    if needed_pages > (free_pages := len(self.free_slots)):
        evicted = self.prefix_cache.evict((needed_pages - free_pages) * self.page_size)
        self.free_slots = torch.cat([self.free_slots, evicted[:: self.page_size]])
        ...
    allocated = self.free_slots[:needed_pages]
    self.free_slots = self.free_slots[needed_pages:]
    return allocated
```

**注**：**RadixPrefixCache.evict** 只负责从树中摘节点并返回物理索引；真正把页放回分配器的是 **CacheManager._allocate**（以及 _free、lazy_free_region）。**free_slots** 始终是 page-aligned 的起始 token 索引。

至此，未命中后缀已经获得了可写的物理位置。forward 完成后，这些位置中的 KV 才具备被复用的条件；因此下一步要看 `cache_req` 如何把它们写回前缀缓存，并同时处理释放与继续持有两种情况。

---

Prefill 前向结束后，用 **cache_req** 把结果写回前缀缓存，并释放不再占用的 page，同时保护仍在使用的前缀不被误删：

```python
# scheduler/cache.py（节选）
def cache_req(self, req: Req, *, finished: bool) -> None:
    insert_ids = req.input_ids[: req.cached_len]
    page_indices = self.page_table[req.table_idx, : req.cached_len]
    old_handle = req.cache_handle
    cached_len, new_handle = self.prefix_cache.insert_prefix(insert_ids, page_indices)
    self.unlock(old_handle)
    self._free(page_indices[old_handle.cached_len : cached_len])  # 已被其他请求写入 prefix cache 的部分，释放避免泄漏
    if finished:
        self._free(page_indices[new_handle.cached_len :])         # 请求结束，释放尾部
    else:
        req.cache_handle = new_handle
        self.lock(new_handle)                                     # 继续保护，留给 decode
```

完整性检查用于尽早发现 page 泄漏，用于即时止损：

```python
def check_integrity(self) -> None:
    self.prefix_cache.check_integrity()
    cache_pages = self.prefix_cache.size_info.total_size // self.page_size
    if len(self.free_slots) + cache_pages != self.num_pages:
        raise RuntimeError(...)
    if self.page_size > 1:
        assert torch.all(self.free_slots % self.page_size == 0)
```

这一步完成了 Prefill 结果的归档，但请求是否能进入 Decode 还取决于它是不是完整的 Req。下面把这条边界单独拎出来，因为 ChunkedReq 正是通过它避免“半成品 Prefill 提前开始采样”的情况。

---

Prefill 全部完成后，请求才有资格进入 Decode。ChunkedReq 重写了 `can_decode`，固定返回 False，因此不会被加入 Decode 管理器；它表示 Prefill 仍在分块处理中，不能提前采样。**只有普通 Req（Prefill 已完成）且 `remain_len > 0`** 才会进入 Decode。换句话说，对单个请求而言，Decode 的起点由 Prefill 的完成状态决定，而不是由某一轮 chunk 是否已经算完决定。

Engine 每次 Decode 前向后调用 **req.complete_one()**：

```python
# core.py
def complete_one(self) -> None:
    self.cached_len = self.device_len
    self.device_len += 1
```

将刚生成的 token 纳入已缓存长度，并为下一个 token 腾出 device_len 空间。

---

### 4.4 Decode、正常结束与取消

进入 Decode 后， **handle** 的职责也发生了变化：它继续锁住 Prefill 已归档的前缀，但中间步骤产生的新后缀暂时只写入该请求自己的 page table。Decode **不会**在每一步都把新后缀插入 Radix Tree；Engine 调用 `req.complete_one()` 推进长度，Scheduler 再通过 `append_host` 把新 token 接到 `input_ids` 上。只有请求结束时，系统才会决定哪些后缀值得归档。

请求**正常结束**时（remain_len == 0 或遇到 EOS），统一释放：

```python
# scheduler/scheduler.py（节选）
def _free_req_resources(self, req: Req) -> None:
    self.table_manager.free(req.table_idx)
    self.cache_manager.cache_req(req, finished=True)
```

cache_req(..., finished=True) 用 input_ids[:cached_len] 与对应 page 索引尝试 insert_prefix：能按页对齐进入树的部分保留为可复用缓存；已被其他请求占用的中间段、以及无法入树的尾部页，会 **_free** 回 free list。

**用户取消**时，Scheduler 先在 prefill / decode 队列里查找：

```python
# scheduler/scheduler.py（节选）
elif isinstance(msg, AbortBackendMsg):
    req_to_free = self.prefill_manager.abort_req(msg.uid)
    req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
    if req_to_free is not None:
        self._free_req_resources(req_to_free)
```

正常结束与取消共用 **_free_req_resources()**，因此两条退出路径最终会收敛到同一套缓存清理逻辑。

处理上一批结果时，释放操作还包在 **lazy_free_region** 里：本批多次 **_free** 会先暂存，退出上下文后再一次性拼回 free_slots。这样既减少中间状态变化，也让下面的批量处理更容易保持一致：

```python
# scheduler/scheduler.py（节选）
with self.cache_manager.lazy_free_region():
    for req in batch.reqs:
        if isinstance(req, ChunkedReq):
            continue  # 半成品 Prefill，不采样、不在此结束
        req.append_host(next_token)
        finished = (not req.can_decode) or (next_token == eos ...)
        if finished and req not in self.finished_reqs:
            self.decode_manager.remove_req(req)
            self._free_req_resources(req)          # cache_req(finished=True)
            new_finished_reqs.add(req)
        elif batch.is_prefill:
            # 非 chunk 的 Prefill 完成：写入前缀缓存，继续留给 Decode
            self.cache_manager.cache_req(req, finished=False)
```

这样既减少频繁拼接小 tensor，也避免 overlap scheduling 下资源状态在处理中途反复变化；**finished_reqs** 则用于防止同一请求被 free 两次。

## 5 对照真实 SGLang

前面的流程说明了 mini-sglang 中一条请求如何完成匹配、分配、写入和释放。真实 SGLang 的模块划分更细，但这些部分仍然可以对应起来：

| 功能 | mini-sglang（`python/minisgl/`） | SGLang（`python/sglang/`） |
|------|----------------------------------|---------------------------|
| Radix / 前缀缓存 | kvcache/radix_cache.py| srt/mem_cache/radix_cache.py；并在向 unified_cache/（Unified Radix Cache）收敛，另有 HiCache 等变体 |
| 请求侧缓存协调（match / lock / 分配 / 释放） | scheduler/cache.py | 分散在 srt/mem_cache/ 与 srt/managers/scheduler.py；Prefill 阶段逻辑多在 srt/managers/schedule_policy.py |
| 调度策略 | 简化 FCFS：按 pending_list 顺序 + 空间不够就 break | srt/managers/schedule_policy.py|
| Kernel | kernel/| 统一入口 kernels/（ops/ 算子、jit/ 编译运行时等）；预编译实现还常走独立包 sgl-kernel |
| KV 物理存储 / 分配 | kvcache/mha_pool.py 等 | srt/mem_cache/pool/ 、srt/mem_cache/storage/|

阅读 SGLang 时，仍可按同一条生命周期追踪：

- **匹配结果在哪生成** —— 前缀树 / Unified Radix（mem_cache）；  
- **何时加锁** —— 调度与 lock_ref / handle 保护；
- **未命中部分在哪分配** —— allocator + pool，对应 mini-SGLang 里的 allocate_paged / free list；
- **完成或取消时由谁写入与释放** —— scheduler 路径上的 cache insert / free，对应 mini-SGLang 的 cache_req / _free_req_resources。

文件更多、介质与策略更复杂，但主生命周期与 mini-sglang 相通。阅读真实 SGLang 时，可以先用 mini-sglang 建立这条主线，再把每个动作映射到更细的模块中。

## 6 总结与测试题


### 6.1 课程总结

一次 Prefix Cache 命中并不是简单地把 KV 张量从 Radix Tree 里取出来。在 mini-sglang 里：

1. **Radix Tree** 按 token 前缀匹配，得到的是写入 page table 的**物理页索引**，真正的 KV 仍在 KV Cache pool 中；
2. 用 **cache handle + lock** 保护这些仍被活动请求使用的节点，避免被淘汰；
3. Scheduler 把命中段的索引与**新分配**的未命中后缀拼进同一张 page table ；
4. free list 不够分配给未命中后缀部分时，从**可淘汰的叶子**开始 evict，由 **CacheManager** 把页放回 free_slots；
5. Prefill 结束或请求结束时，**cache_req** 把可入树的前缀写回 Radix Tree；已被别人占用的中间段、以及无法对齐入树的尾部则 _free。

---

### 6.2 测试题


1. 为什么 **CacheManager.match_req** 只传入 **input_ids[:input_len - 1]**？如果允许整个 prompt 全部命中，模型前向会遇到什么问题？
> 提示：Prefill 既要写出本轮要算的 KV Cache，也要产出**第一个输出 token 的 logits**。


2. 为什么 **PrefillAdder._try_allocate_one** 要在 lock(handle) 前后各检查一次 available_size？
> 提示：两次检查都把 estimated_len + reserved_size（含 Decode 预留、以及为 decode 的预占用空间）和 available_size 比较；中间的 lock 会改变可淘汰规模。        


3. **RadixCacheHandle.get_matched_indices** 为什么要沿父节点走到根后再反转拼接？

> 提示：每个树节点只存本段 value，从匹配节点沿 parent 向上。


4. 在 cache_req 中，**page_indices[old_handle.cached_len:cached_len]** 对应哪种场景？不释放会造成什么后果？

> 提示：本请求 Prefill 时为未命中后缀部分分配页；insert_prefix 时发现其中一段已被其它请求写入树，返回的 cached_len 大于 old_handle.cached_len。


5. 为什么 **evict** 必须从无引用叶子开始，谁负责把页真正放回 free list？

> 提示：只考虑在 mini-SGLang 中的情况，可以参考 mini-SGLang 中 Radix Cache 相关实现代码部分。

## 参考资料

- [mini-sglang](https://github.com/sgl-project/mini-sglang)
- [SGLang RadixAttention](https://lmsys.org/blog/2024-01-17-sglang/)
- [SGLang](https://github.com/sgl-project/sglang)

