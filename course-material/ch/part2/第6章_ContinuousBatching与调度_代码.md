# 第 6 章 连续批处理与调度（代码实现）

HTTP 服务可以同时接收很多请求，但模型前向仍然需要回答一个具体问题：这一轮算谁？如果把一组请求一直绑到最长的那个结束，短请求腾出的空间就会空着。本章实现一个按迭代重新组 batch 的调度器，让新请求进入、已完成请求退出，并把每条请求的 KV Cache 与输出对应起来。

本篇对应[概念篇](./第6章_ContinuousBatching与调度.md)，完整代码在 `examples/part2/scheduler.py`，复用[第 3 章](./第3章_前向与生成_代码.md)的模型与缓存接口。调度策略参考 mini-sglang [9a91cfa](https://github.com/sgl-project/mini-sglang/tree/9a91cfafe754aa85daee49998176275667eb58f2)：有可执行的 prefill 时先做 prefill，否则做 decode。教学版使用 CPU 上的普通张量，不需要启动 HTTP 服务或下载权重。

## 1 本章学习目标

完成本章后，你将能够：

1. 实现 waiting、running、finished、cancelled 请求状态及其转换。
2. 在每轮调度时同时检查请求数、prefill token 预算和容量预留。
3. 将不同长度请求的 KV 打包成一次模型前向，再按请求拆回。
4. 正确处理首 token、EOS、长度结束、取消和后端异常。
5. 用调度轨迹和串行基线判断实现是否正确，并区分逻辑轮次与真实延迟。

## 2 运行并读懂一条轨迹

### 2.1 启动实验

按第 3 章安装 `examples/part2/requirements.txt` 后，在仓库根目录执行：

```bash
python -m examples.part2.scheduler
python -m pytest examples/part2/tests -q -k scheduler
```

默认模型为随机小型 Qwen3，采样使用贪心。A、B 在开始时进入，C 在第一轮完成后到达；最多同时运行两条请求。A、B、C 的最大新增 token 数分别是 2、5、2，未设置 EOS，因此只由长度结束。

```text
tick=0 prefill batch=['A', 'B'] reserved_after=12
tick=1 decode  batch=['A', 'B'] reserved_after=8
tick=2 prefill batch=['C'] reserved_after=11
tick=3 decode  batch=['B', 'C'] reserved_after=8
tick=4 decode  batch=['B'] reserved_after=8
tick=5 decode  batch=['B'] reserved_after=0
All request caches released: True
```

`tick` 表示发生了一次模型前向，不是毫秒。`reserved_after` 是本轮结束后仍被活动请求预留的 token 容量，不是本轮输入量，更不是已经测量的显存占用。

### 2.2 每轮究竟发生什么

| 轮次 | 本轮输入 | 本轮结束后的变化 |
| --- | --- | --- |
| 0 | A、B 的完整 prompt | 二者各产生第一个 token，建立 KV |
| 1 | A、B 上轮产生的 token | A 达到长度上限并释放，B 留下 |
| 2 | C 的完整 prompt | C 填入空位，B 暂停一轮 |
| 3 | B、C 各一个 token | C 结束，B 继续 |
| 4、5 | B 每轮一个 token | B 最终结束，全部容量归还 |

固定 batch 通常要等同组最长请求结束才接入下一组；这里 A 退出后，C 在 B 完成前就开始了。这才是本例展示的动态行为。它不要求每轮一定凑满 batch，也不意味着 prefill 和 decode 必须出现在同一个 batch。

## 3 请求状态与资源归属

### 3.1 Request 保存哪些东西

`Request` 保留 uid、prompt、生成上限、EOS、已生成输出、状态、结束原因、KV，以及首 token 和完成的逻辑轮次。它不持有 HTTP 连接；调用方可以用 `Scheduler.step()` 返回的 `(uid, token, finish_reason)` 事件，把结果送回对应连接。

一条请求的正常状态路径是 waiting → running → finished。waiting 或 running 都可以进入 cancelled。终态请求不会再次组 batch。未知 uid 或已结束 uid 的取消返回 `False`，不重复释放。

`max_new_tokens=0` 在提交时直接完成，输出为空，不占活动槽位，也不调用模型。非法输入和永远无法容纳的请求在进入队列之前报错，避免产生永远等不到资源的队首。

### 3.2 三种预算不能混淆

| 参数 | 控制对象 | 本例规则 |
| --- | --- | --- |
| `max_running` | 活动请求条数 | 达到上限后，新请求等待 |
| `prefill_budget` | 一轮新接入 prompt 的 token 总数 | 本轮超预算则停止接入 |
| `token_capacity` | 活动请求的容量预留总额 | 每条预留 `prompt_len + max_new_tokens` |

预留完整最大长度是一种保守策略：少接入一些请求，换取运行过程中不会因为这份逻辑预算耗尽而突然无路可走。本例的容量只是 token 级记账，没有向 CUDA 分配固定大小的页池；实际内存还包含参数、激活、打包 KV 和临时张量，因此不能把 `token_capacity` 当成严格的显存上限。

例如 A 的 prompt 长度为 2、最大输出为 2，预留 4；B 的 prompt 长度为 3、最大输出为 5，预留 8。A 结束后归还 4，B 仍占 8。已经结束的 token ID 列表保留在 `requests` 中供查看，KV 则立即置空。长期服务还需要消费并清理历史结果，不能无限保留所有请求记录。

### 3.3 一个必须始终成立的不变量

每轮完成后应满足：

```python
self.reserved == sum(r.reservation for r in self.running.values())
```

所有资源释放都经过 `_finish`：活动请求先从 `running` 删除，再减预留量、清空 KV，记录状态与原因。集中出口让长度结束、EOS、取消和异常能够使用相同的记账规则。重复调用 `_finish` 会被终态检查拦住。

## 4 写出每轮调度

### 4.1 接入阶段：严格 FIFO

`submit` 把请求加入 `deque`。`step` 从队首尝试接入，只有请求槽位、容量和本轮 prefill 预算全部满足才继续。如果队首暂时无法接入，本轮不绕过它选择更小的后续请求。

这是刻意选择的严格 FIFO，容易解释和测试，但会有队首阻塞。当前实现不支持 chunked prefill，所以 prompt 本身超过 `prefill_budget` 时直接拒绝。不能把它留在 waiting 后每轮重试，因为它永远不会满足条件。

### 4.2 选择 prefill 或 decode

调度核心可以缩成两行：

```python
phase = "prefill" if admitted else "decode"
batch = admitted or list(self.running.values())
```

若本轮接入新请求，则只执行这些新请求的 prefill；已有 running 请求保留状态，等待下一轮。否则，对当前全部 running 请求各推进一个 token。后端一次返回一个 token 对应一个 batch 元素，调度器依次提交输出、判断终止、产生事件。

没有可执行请求时，`step` 返回空列表。本例由调用方控制轮次，没有网络接收循环；真实服务空闲时应阻塞等待消息，避免空转占用 CPU。

### 4.3 公平性和性能取舍

prefill 优先可以较早给新请求首 token，但连续的新请求接入可能推迟已有请求的下一 token，甚至在持续负载下造成 decode 饥饿。decode 优先则可能延长新请求等待时间。增加 chunked prefill、限制连续 prefill 轮数或混合 batch，都需要把预算和公平性一起设计。

本章实现固定策略，方便读者验证状态机。改变策略后首先验证输出与资源不变量，再在相同请求集、到达时间和硬件环境下测量 TTFT、ITL、吞吐与尾延迟。不能根据“循环次数更少”直接推断服务更快，因为不同轮次的工作量差别很大。

## 5 让不同长度请求真正一起前向

### 5.1 Prefill：补齐输入，但不缓存 padding

`KVEngine.step` 在 prefill 阶段把 prompt 右侧补齐到本 batch 最长长度，同时建立布尔 `attention_mask`。一次模型调用得到所有请求的 logits 和 KV。

右补齐后，不能统一读取 `logits[:, -1]`，因为短请求的最后一列是 padding。代码按每条请求的 `prompt_len - 1` 取 logits。随后只选取有效位置，把该请求的每层 KV 保存到 `Request.cache` 中。随机 padding ID 不会参与有效 query 的注意力。

### 5.2 Decode：每条请求只输入一个 token

decode 时，一条请求的输出末尾是上轮已采样、尚未写入 KV 的 token。本轮 `input_ids` 为 `[B,1]`。历史 KV 长度可能不同，因此先把每层 K/V 打包到 `[B,Hkv,max_past_len,D]`，短请求的空位为零，但必须被 mask 遮住。

假设 B、C 的有效历史长度分别为 4、1，本轮的 KV 有效性如下：

```text
              历史打包区域      本轮输入
B mask        1 1 1 1           1
C mask        1 0 0 0           1
B 逻辑位置                      4
C 逻辑位置                      1
```

C 的新 token 物理上也追加到索引 4，但其逻辑位置是 1。第 3 章模型根据 mask 的有效 token 累计数产生 RoPE 位置，因此不会把 KV 打包的空洞误认为真实上下文。

一次前向结束后，通过 `mask[i].nonzero()` 取回每条请求的有效 KV，压紧后分别存储。没有请求间共享可变 KV；组 batch 和拆 batch 都保持 uid 与行号对应。

### 5.3 为什么它仍然只是教学后端

这条路径确实做了一次批量模型调用，也确实在 decode 复用了 KV；并不是对每条请求逐个调用 `generate`。但它每轮还会打包、复制、扩展和压紧张量，成本很高。

生产引擎通过页表和预分配存储，让各请求的 KV 长期留在缓存池中；注意力内核根据元数据读取分散位置，避免这里的反复复制。[第 7 章](./第7章_PagedKVCache与显存管理.md)接着解决这个问题。动态 batch 是请求调度策略，分页是存储组织方式，两者配合但不是同一个概念。

## 6 结束、取消与验证

### 6.1 终止发生在 token 提交之后

先把本轮生成 token 加入输出，再检查 EOS 和长度。EOS 本身计入输出，并返回 `finish_reason="eos"`；普通长度结束返回 `"length"`。这与第 3 章的单请求接口一致。

本例在 `step` 之间同步取消，取消时没有 GPU kernel 在执行，所以可以立即移除和清理。若模型调用抛出异常，或返回 token 数量与 batch 不一致，本批请求统一以 `"error"` 结束并归还预留，随后把异常交回调用方；本例没有实现故障重试和错误网络响应。

真实异步引擎中的取消需要考虑正在执行的 batch：可以先标记请求取消、抑制后续输出，等引用该资源的执行结束后再回收。把本例的立即释放原样用在仍被 GPU 使用的缓存上会产生生命周期错误。

### 6.2 与串行基线逐条比较

`test_continuous_batch_matches_serial_and_reuses_slot` 对 A、B、C 分别运行第 3 章的独立 `generate`，确认输出与动态 batch 完全一致，并检查 C 在 B 完成前获得首 token。它还检查 decode 输入形状是 `[2,1]`，防止不小心退回完整前缀重算。

其余测试覆盖 waiting/running 取消、重复取消、EOS、零输出、预算阻塞后继续推进、永久超限拒绝和后端异常清理。最终必须满足活动请求为空、预留归零、所有终态请求的 KV 为空。测试中的 tick 仅用于验证调度顺序，没有被报告成硬件性能。

### 6.3 完整程序

课程站点直接引入可执行源码；在 GitHub Markdown 中请打开 `examples/part2/scheduler.py`。

::: details 展开 scheduler.py

<<< @/examples/part2/scheduler.py

:::

## 7 对照真实 mini-sglang

固定提交中可沿着如下路径阅读，先只追正常循环，再读 overlap：

| 教学接口 | mini-sglang 9a91cfa 中的位置 | 阅读问题 |
| --- | --- | --- |
| `submit` | `Scheduler._process_one_msg` | `UserMsg` 如何转入 prefill 管理器？ |
| 接入检查 | `PrefillAdder.try_add_one`、`_try_allocate_one` | token 预算、请求表、KV 容量如何共同限制接入？ |
| prefill 组批 | `PrefillManager.schedule_next_batch` | chunked 请求如何保留在 pending 队列？ |
| decode 组批 | `DecodeManager.schedule_next_batch` | 为什么固定 uid 顺序对多 rank 一致性有帮助？ |
| 选择下一轮 | `Scheduler._schedule_next_batch` | 为什么 `prefill or decode` 会产生优先级？ |
| 后端计算 | `Scheduler._prepare_batch`、`_forward` | 位置、页表、采样参数怎样进入 Engine？ |
| 输出和回收 | `Scheduler._process_last_data`、`_free_req_resources` | 结束标志和异步计算如何影响回收时机？ |

`normal_loop` 是本章容易对应的入口；`overlap_loop` 会重叠当前 batch 执行与上一批结果处理。代码里对重复释放的保护正说明：开始重叠执行后，“这一轮结束”的直观定义不再足以描述所有在途引用。

该版本还支持 chunked prefill 和前缀缓存。教学版保留了策略主线，省去了 CUDA stream、ZMQ、多 rank 同步、分页和共享前缀锁。读源码时应分别追问调度选择、张量布局、缓存所有权，避免把所有内容都归结为一个 while 循环。

## 8 总结与测试题

### 8.1 课程总结

连续批处理在每轮重新决定参与计算的请求。我们实现了请求接入、独立缓存、prefill/decode 组批和统一回收，并通过串行生成基线验证输出语义。容量预留、请求条数和本轮 token 预算分别解决不同问题；正确调度的前提是所有终止路径都维护资源不变量。

### 8.2 测试题

1. 默认例子中，为什么 tick 2 只有 C，B 没有参与？这会影响哪项延迟指标？
2. 如果把右 padding 的短请求也从最后一列 logits 采样，会发生什么？
3. 同一个 decode batch 的两条请求，物理 KV 长度与逻辑 token 位置为什么可能不同？
4. 为什么本例拒绝 prompt 长度超过 prefill 预算的请求？怎样用 chunked prefill 改进？
5. 如果启用异步 GPU 执行，取消时还需要等待哪些资源使用者？

## 参考资料

- [mini-sglang 9a91cfa：Scheduler 主循环](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/scheduler.py)
- [mini-sglang 9a91cfa：PrefillManager 与 PrefillAdder](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/prefill.py)
- [mini-sglang 9a91cfa：DecodeManager](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/decode.py)
- [mini-sglang 9a91cfa：Req、Batch 与 Context](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/core.py)
