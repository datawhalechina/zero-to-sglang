# 第 6 章 连续批处理与调度（代码走读）

A、B 正在生成时，C 又来了。A 比 B 先结束，C 怎样用上空出来的位置？本篇承接[概念篇](./第6章_ContinuousBatching与调度.md)，沿 mini-sglang 的接收、组批、执行与回收路径，观察 batch 如何在轮次之间变化。

本文固定使用参考提交 [9a91cfa](https://github.com/sgl-project/mini-sglang/tree/9a91cfafe754aa85daee49998176275667eb58f2)。代码块是源码节选，简化处会明确标注；它们依赖原文件中的类和运行环境，不能作为独立调度器直接运行。先读同步的 `normal_loop`，再理解重叠执行需要额外保存的状态。

## 1 本章学习目标

完成本章学习后，你将能够：

1. 找到等待队列、运行集合及它们之间的转换。
2. 对照源码解释 prefill 优先、请求槽位与 token 预算。
3. 追踪输出写回、EOS、长度上限和资源回收。

## 2 定位源码

以下路径均相对于 mini-sglang 的 `python/minisgl/`：

| 文件与入口 | 阅读重点 |
| --- | --- |
| `scheduler/scheduler.py`：`Scheduler.normal_loop` | 一轮接收、组批、计算与结果处理 |
| `scheduler/prefill.py`：`PrefillManager`、`PrefillAdder` | 等待请求的接入与预算检查 |
| `scheduler/decode.py`：`DecodeManager` | 继续生成的请求集合 |
| `core.py`：`Req` | 已缓存长度、生成余量与请求状态 |
| `scheduler/table.py`、`scheduler/cache.py` | 请求槽位与缓存资源 |

HTTP 与分词完成后，调度器收到的是 `UserMsg`。`_process_one_msg` 检查模型长度限制，必要时缩短最大输出长度，再调用 `prefill_manager.add_one_req` 加入等待队列。

## 3 从主循环看一轮调度

### 3.1 接收、组批、执行、处理结果

`normal_loop` 按下面的顺序执行：

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

有请求可处理时，接收消息不会阻塞本轮计算；没有工作时，才等待新消息。`ongoing_data` 把本轮的输入和输出放在一起，结果处理才能知道每个 token 属于哪条请求。

### 3.2 先尝试 prefill，否则 decode

`_schedule_next_batch` 的选择逻辑为：

```python
# scheduler/scheduler.py: Scheduler._schedule_next_batch
batch = (
    self.prefill_manager.schedule_next_batch(self.prefill_budget)
    or self.decode_manager.schedule_next_batch()
)
return self._prepare_batch(batch) if batch else None
```

这里的 `or` 是策略：能组出 prefill batch 就优先执行，否则推进 decode。**等待队列非空，不代表一定能组出 prefill batch**；资源不足时仍可能转去 decode。`_prepare_batch` 随后分配所需 KV 页、构造位置与映射，并准备 Attention 元数据。

连续批处理不要求每轮把新旧请求混在同一个 batch 中。这个版本会在不同轮次选择 prefill 或 decode，活动请求集合仍然可以持续变化。

## 4 请求怎样进入与留在 batch 中

### 4.1 等待队列按顺序尝试接入

`PrefillManager.pending_list` 按到达顺序保存请求。下面节选只展示接入顺序，省略 chunked prefill 的续块记录：

```python
# scheduler/prefill.py: PrefillManager.schedule_next_batch (simplified)
reqs = []
for pending_req in self.pending_list:
    if req := adder.try_add_one(pending_req):
        reqs.append(req)
    else:
        break
```

队首接入失败就停止本轮尝试，不绕过它寻找更小的请求。原实现还会把未完成 prefill 的请求放回队首，下一轮继续其剩余 prompt。

### 4.2 三种资源约束不要混为一谈

`PrefillAdder` 分别检查以下约束：

| 约束 | 源码中的检查 | 含义 |
| --- | --- | --- |
| 请求槽位 | `table_manager.available_size` | 是否还有请求表行可分配 |
| 本轮计算预算 | `token_budget` | 本轮还能处理多少个 prefill token |
| KV 容量估计 | `estimated_len + reserved_size` | 为新请求及运行中的请求预留多少 token 空间 |

首次接入时，容量估计的核心是：

```python
# scheduler/prefill.py: PrefillAdder._try_allocate_one (excerpt)
extend_len = req.input_len - cached_len
estimated_len = extend_len + req.output_len
if estimated_len + self.reserved_size > self.cache_manager.available_size:
    return None
```

`cached_len` 来自前缀匹配；`reserved_size` 初始来自 `decode_manager.inflight_tokens`，还要为本轮已接入请求累加。缓存句柄加锁后，可回收容量可能变化，源码会再次检查容量。

长 prompt 不必一次算完：`_add_one_req` 使用 `min(token_budget, remain_len)` 选择本轮片段。若还有剩余，就生成 `ChunkedReq`；中间片段不提交输出 token，也不进入 decode 集合。这里的预算以 token 数量估算，不能直接当成显存字节数。

### 4.3 保留还需要生成的请求

`_forward` 调用引擎后，通过 `DecodeManager.filter_reqs` 更新运行集合：

```python
# scheduler/decode.py: DecodeManager.filter_reqs
self.running_reqs = {
    req for req in self.running_reqs.union(reqs) if req.can_decode
}
```

`Req.can_decode` 表示还有长度余量；EOS 需要等采样结果回传后再判断。下一次 decode 会把运行集合按 uid 排序组成 batch，每条请求推进一个 token。请求输出始终按各自的 uid、表行和长度保存。

## 5 输出写回与动态组批

### 5.1 完成请求，归还资源

`_process_last_data` 先等待 token 从 GPU 复制到 CPU，再按 `batch.reqs` 的顺序处理结果。以下节选省略回复消息、chunk 检查和重叠执行的防重复释放逻辑：

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

`_free_req_resources` 归还请求表行，并调用 `cache_manager.cache_req(req, finished=True)` 处理缓存。结束不等于立即删除所有物理 KV：可复用前缀可能继续保留在缓存中，具体生命周期见第 7、8 章。

取消消息走 `_process_one_msg` 的 `AbortBackendMsg` 分支：先从等待队列或运行集合移除请求，有已分配资源时再调用同一个回收入口。尚未分配资源的排队请求只需移出队列。重叠执行还有在途计算，阅读时应连同同步与防重复释放逻辑一起检查。

### 5.2 手算 A、B、C 的变化

假定只有两个请求槽位、KV 容量充足、没有前缀命中，三个 prompt 都能在一轮 prefill 中完成。A/B/C 分别需要生成 2/5/2 个 token，均不会提前遇到 EOS；C 在第 0 轮结束后到达。prefill 完成时就产生第一个输出 token。

| 轮次 | 本轮 batch | 轮末变化 |
| --- | --- | --- |
| 0 | prefill：A、B | A、B 各得到第 1 个 token；随后 C 到达 |
| 1 | decode：A、B | A 得到第 2 个 token 后结束，释放槽位 |
| 2 | prefill：C | C 使用空位；B 本轮暂停 |
| 3 | decode：B、C | C 得到第 2 个 token 后结束 |
| 4 | decode：B | B 得到第 4 个 token |
| 5 | decode：B | B 得到第 5 个 token 后结束 |

C 不必等 B 结束就能接入，这就是连续批处理在此处的具体表现。表格是按上述假设推演的状态变化，不是服务日志或耗时测量。持续优先执行 prefill 也可能推迟已有请求的 decode，公平性需要结合真实负载评估。

## 6 总结与测试题

### 6.1 课程总结

`normal_loop` 把接收、组批、执行和结果处理串起来。PrefillManager 决定谁能进入，DecodeManager 保留继续生成的请求，结果处理负责结束判断与资源回收。连续批处理的关键是每轮更新这些状态，让新请求复用已经空出的容量。

### 6.2 测试题

1. 为什么等待队列非空时，仍可能执行 decode？
2. 请求槽位充足但 KV 容量不足时，请求能否进入？应查看哪段代码？
3. 为什么 chunked prefill 的中间片段不能提交生成 token？
4. 在第 5.2 节的假设下，若 A 只生成 1 个 token，C 最早在哪一轮开始？
5. 正常结束和取消分别从哪里进入资源回收？为什么释放请求槽位不等于删除全部 KV？

## 参考资料

- [mini-sglang：调度主循环](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/scheduler.py)
- [mini-sglang：Prefill 接入](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/prefill.py)
- [mini-sglang：Decode 运行集合](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/decode.py)
- [mini-sglang：请求状态](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/core.py)
- [mini-sglang：请求表](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/table.py)
- [mini-sglang：缓存管理](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/cache.py)
- [第 7 章：Paged KV Cache 与显存管理](./第7章_PagedKVCache与显存管理.md)
- [第 8 章：RadixAttention 与前缀缓存](./第8章_RadixAttention与前缀缓存.md)
