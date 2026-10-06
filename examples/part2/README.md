# Part II：第 3、6、10 章配套代码

本目录包含三个可以在 CPU 上运行的教学程序。先完成第 3 章，再读第 6、10 章；三个程序共享同一个手写 Qwen3 模型。

| 章节 | 程序 | 中文教程 |
| --- | --- | --- |
| 第 3 章 | `forward.py` | [前向与生成](../../course-material/ch/part2/第3章_前向与生成_代码.md) |
| 第 6 章 | `scheduler.py` | [连续批处理与调度](../../course-material/ch/part2/第6章_ContinuousBatching与调度_代码.md) |
| 第 10 章 | `speculative.py` | [投机解码](../../course-material/ch/part2/第10章_投机解码_代码.md) |

## 安装与运行

所有命令在仓库根目录执行。推荐 Python 3.12。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r examples/part2/requirements.txt

python -m examples.part2.forward
python -m examples.part2.forward --cache
python -m examples.part2.scheduler
python -m examples.part2.speculative
python -m examples.part2.speculative --temperature 1
python -m pytest examples/part2/tests -q
```

默认全部使用随机小模型，不需要网络、模型权重或 GPU。输出 token IDs 用于验证实现，没有自然语言质量含义。依赖下载需要网络；安装后，测试与默认示例均可离线执行。

## 可选：加载预训练 Qwen3

```bash
python -m examples.part2.forward \
  --model Qwen/Qwen3-0.6B \
  --prompt "用一句话解释 KV Cache。" \
  --max-new-tokens 32 --cache
```

支持模型 ID 或本地模型目录。使用 CPU FP32，参数本身约需 2.4 GB，另需激活和加载过程内存。模型须为未启用 RoPE scaling、滑窗或 Attention bias 的 dense Qwen3；不支持的配置会被拒绝。Transformers 只负责权重和 tokenizer 加载，前向和生成使用本目录代码。需要固定 checkpoint revision 时，在 Python 中调用 `runtime.load_pretrained(path, revision=...)`。

## 验证覆盖

- 前向：同权重 Transformers logits 对齐、因果性、分块/逐 token KV 对齐、权重共享、本地 checkpoint 加载。
- 生成：贪心、温度及 top-k/top-p、EOS、零输出、最大长度。
- 调度：变长 prefill/decode、动态接入、与串行基线一致、预算与超限、waiting/running 取消、后端失败回收。
- 投机：多种草稿长度、全接受与拒绝、EOS/bonus、贪心一致、正残差恒等式、已知 Markov 联合分布抽样检验。

概率测试固定 seed，做 6000 次两 token 实验；它是回归检查，分布保持的理由见教程中的推导。不同算法消费随机数的顺序不同，随机采样不以逐字符串一致为标准。

## 教学边界

`scheduler.py` 真正执行批量模型前向并复用 KV，但使用打包、复制和压紧的密集张量；token 容量是保守记账，不是物理显存页池。未包含 HTTP、chunked prefill、分页、CUDA Graph、多卡和异步回收。

`speculative.py` 实现线性草稿与概率校正，用完整前缀重算隔离未接受 token；未实现投机 KV 回滚、树状验证或 EAGLE。统计目标调用次数用于观察算法，不能据此报告真实吞吐加速。

mini-sglang 源码对照固定在 `9a91cfafe754aa85daee49998176275667eb58f2`。投机源码延伸使用 SGLang `1291f3175d3b7614821884e9653985fcd3c3dae8`。预训练权重和 GPU 性能需在相应环境单独验证。
