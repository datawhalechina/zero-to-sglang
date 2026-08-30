<!-- TODO: 替换为项目 banner 图，路径建议 docs/public/zero-to-sglang.png -->
<div align='center'>
    <img src="./docs/images/zero_to_sglang.png" alt="zero-to-sglang banner" width="100%">
    <h1>zero-to-sglang</h1>
    <h3>📚 《从零手搓SGLang》</h3>
</div>

<div align="center">
  <img src="https://img.shields.io/github/stars/datawhalechina/zero-to-sglang?style=flat&logo=github" alt="GitHub stars"/>
  <img src="https://img.shields.io/github/forks/datawhalechina/zero-to-sglang?style=flat&logo=github" alt="GitHub forks"/>
  <img src="https://img.shields.io/badge/language-Chinese-brightgreen?style=flat" alt="Language"/>
  <a href="https://github.com/datawhalechina/zero-to-sglang"><img src="https://img.shields.io/badge/GitHub-Project-blue?style=flat&logo=github" alt="GitHub Project"></a>
  <!-- TODO: 部署在线阅读后启用下面这行，并确认地址 -->
  <a href="https://datawhalechina.github.io/zero-to-sglang/"><img src="https://img.shields.io/badge/在线阅读-Online%20Reading-green?style=flat&logo=gitbook" alt="Online Reading"></a>
</div>

<div align="center">
  <p><em>从推理的本质出发，手搓一个 mini-sglang，再吃透真实 SGLang 的源码</em></p>
  <p>Datawhale × RadixArk</p>
</div>

---

## 🎯 项目介绍

&emsp;&emsp;大模型已经从「训练」卷到「推理」了。应用爆发之后，推理成本和延迟成了真正的瓶颈：同样一块卡，好的推理引擎能多扛好几倍的请求。但市面上讲推理引擎的教程不多，要么只讲概念不动手，要么直接让你啃 SGLang 源码，从哪儿下手都不知道。

&emsp;&emsp;这门课想补上这个缺口。前半部分，我们从推理最核心的问题讲起：KV Cache 为什么要有、prefill 和 decode 差在哪、compute-bound 和 memory-bound 到底是什么意思。中间部分，带着你从零手搓一个 mini-sglang：前向、生成、KV Cache、HTTP 服务、Continuous Batching、Paged KV Cache、RadixAttention，一样一样加上去。后半部分，回到真实的 SGLang，讲清楚它那些前沿优化具体是怎么做的，最后教你怎么给 SGLang 提第一个像样的 PR。

&emsp;&emsp;本项目由 <strong>Datawhale</strong> 和 <strong>RadixArk</strong>（SGLang 团队创立的公司）共同发起，值得信赖。

## ✨ 你能学到什么

- 📖 <strong>读懂推理</strong>：KV Cache、prefill/decode、compute-bound vs memory-bound，这些概念到底在说什么
- 🏗️ <strong>手搓引擎</strong>：从 0 到 1 写出一个 mini-sglang，每一步都落在代码上
- 🛠️ <strong>看懂真实引擎</strong>：RadixAttention、Paged KV Cache、Continuous Batching 这些 SGLang 核心是怎么实现的
- ⚙️ <strong>摸到前沿</strong>：量化、分层缓存、DP Attention / EP / PP、Prefill-Decode 分离
- 🚀 <strong>参与开源</strong>：学会 profiling 和 trace 分析，走通 SGLang 的 PR 全流程

## 📋 前置要求

- **Python 编程**：熟练使用 Python，有基本的软件工程习惯
- **深度学习基础**：熟悉 PyTorch，了解神经网络的基本原理
- **数学基础**：线性代数、概率统计，知道矩阵运算和注意力机制的大致计算即可
- **GPU 编程（可选）**：了解 CUDA 基础概念会更好，不懂也没关系，Part I 会从零补齐
- **硬件（可选）**：Part I 不需要 GPU；Part II 大部分内容能在 CPU 上调试，完整的实现和性能测试建议用 GPU（云服务也行）

## 📖 课程目录

> 状态图例：✅ 已完成  🔄 更新中  📝 待完善  🚧 筹备中  ⏸️ 暂缓
>
> 标注 **粗体** 的章节由 SGLang 官方成员编写。

| 章节 | 关键内容 | 状态 |
|------|----------|------|
| <strong>Part 0 — 开课之前</strong> | | |
| [0.1 编码伦理与开源精神](docs/part0/Part0-编码伦理与开源精神.md) | 对自己代码负责、沟通时说人话、Profile 永远是第一步、开源精神 | ✅ |
| <strong>Part I — 基础概念（concepts only，无代码、无 GPU）</strong> | | |
| 1. Introduction to LLM | LLM 的定义与发展脉络、Transformer 架构、自回归生成、关键基础概念 | 🔄 |
| 2. Introduction to inference | 训练 vs 推理、prefill/decode、compute-bound vs memory-bound、Roofline | 🔄 |
| 3. Introduction to GPU | GPU 架构基础、LLM 推理在 GPU 上的执行流程、从硬件理解推理瓶颈 | 🔄 |
| 4. KV Cache: The Core Data Structure of Inference | 从 Attention 推导 KV Cache、cache 生命周期、显存占用定量分析 | 🔄 |
| [5. Introduction to Benchmark](docs/part1/第5章_Benchmark入门.md) | TTFT / TPOT / ITL / Goodput 等核心指标、百分位与尾延迟、怎么设计/跑/读 benchmark | ✅ |
| <strong>Part II — 从零手搓 Mini SGL</strong> | | |
| 1. mini-sglang：推理引擎长什么样 | 推理引擎的总体架构、模块划分、本部分的 roadmap | 🚧 |
| 2. **Inside SGLang: The Path of a Request** | 一个请求从进入到返回的完整生命周期 | 🚧 |
| 3. Your First 200 Lines: Forward Pass and Generation | 手写前向传播与自回归生成循环 | 🚧 |
| 4. KV Cache: From O(n²) to O(n) | 缓存实现与注意力计算优化 | 🚧 |
| 5. Serving It: HTTP and Concurrent Requests | HTTP 服务化、并发请求处理 | 🚧 |
| 6. Continuous Batching and the Scheduler | 连续批处理与调度器设计 | 🚧 |
| 7. Paged KV Cache and Memory Management | 分页 KV Cache 与显存管理 | 🚧 |
| 8. RadixAttention and Prefix Caching | RadixAttention 与前缀缓存 | 🚧 |
| 9. Multi-process & Tensor Parallelism | 多进程与张量并行 | 🚧 |
| 10. Speculative Decoding | 投机解码 | 🚧 |
| <strong>Part III — 高级推理技术（深入真实 SGLang）</strong> | | |
| 1. **Attention Backends（FlashInfer / Triton / FA3 / FlashMLA）and CUDA Graph** | 主流 Attention 后端对比与 CUDA Graph | 🚧 |
| 2. **Quantization and Low-Precision Inference** | 量化与低精度推理 | 🚧 |
| 3. **Hierarchical Caching** | 分层缓存 | 🚧 |
| 4. **Scaling Out: DP Attention, EP, PP** | 横向扩展：DP Attention / EP / PP | 🚧 |
| 5. **Prefill-Decode Disaggregation** | Prefill-Decode 分离 | 🚧 |
| <strong>Part IV — 如何为 SGLang 做贡献（可选）</strong> | | |
| 1. **Deploying SGLang with the Cookbook** | 使用 Cookbook 部署 SGLang | 🚧 |
| 2. **Measuring It All: Profiling and Trace Analysis** | Profiling 与 Trace 分析 | 🚧 |
| 3. **SGLang PR Workflow** | SGLang PR 全流程 | 🚧 |

## 🗓️ 编写进度

起始日期：2026 年 8 月 24 日（周一）。每章约一周，加粗章节由 SGLang 官方成员编写。

| 部分 | 时间安排 | 状态 |
|------|----------|------|
| Part 0 — 开课之前 | 已完成 | ✅ |
| Part I — 基础概念 | 8.24 ~ 9.06 编写，9.07 ~ 9.13 review | 🔄 |
| Part II — 从零手搓 Mini SGL | 9.14 ~ 10.04 编写，10.05 ~ 10.10 review | 🚧 |
| Part III — 高级推理技术 | 10.10 ~ 11.15 | 🚧 |
| Part IV — 如何为 SGLang 做贡献 | 11.16 ~ 12.6 | 🚧 |

## 🚀 快速开始

```bash
# 克隆仓库
git clone https://github.com/datawhalechina/zero-to-sglang.git
cd zero-to-sglang
# 安装基础依赖（根据具体章节需求安装）
```

### 学习路径

1️⃣ 读 Part I，先把推理的整体认知建立起来（不需要 GPU）  
2️⃣ 跟着 Part II，从 0 到 1 手搓一个 mini-sglang  
3️⃣ 进 Part III，对照真实 SGLang 源码理解前沿优化  
4️⃣ 完成 Part IV，走通 profiling、trace 分析和 PR 全流程

### 项目结构

```
zero-to-sglang/
├── docs/                    # 课程文档
│   ├── part0/               # Part 0：编码伦理与开源精神
│   ├── part1/               # Part I：基础概念（更新中）
│   ├── part2/               # Part II：从零手搓 Mini SGL（规划中）
│   ├── part3/               # Part III：高级推理技术（规划中）
│   └── part4/               # Part IV：如何为 SGLang 做贡献（规划中）
├── README.md                # 项目说明
└── .gitignore               # Git 忽略配置
```

<!-- TODO: 若后续有配套代码目录，请补充到上面的结构树中 -->

## 💡 如何学习

&emsp;&emsp;这门课理论和实战对半分，适合有 Python 基础、对 LLM 有个大致概念的人。不需要你懂 CUDA，也不用一上来就有 GPU。

&emsp;&emsp;五个部分按顺序走就行：

- <strong>Part 0：开课之前</strong>。先聊点技术之外的：怎么对自己写的代码负责、怎么和 reviewer 沟通、为什么改性能之前必须先 profile。这部分很短，但值得先看。

- <strong>Part I：基础概念</strong>（第 1 ~ 5 章）。没有代码，也不需要 GPU。把 LLM、推理、GPU、KV Cache、benchmark 这几个概念掰开讲清楚，建立心智模型。

- <strong>Part II：从零手搓 Mini SGL</strong>（第 1 ~ 10 章）。动手。从 200 行前向和生成开始，一步步把你的 mini-sglang 搭起来。

- <strong>Part III：高级推理技术</strong>（第 1 ~ 5 章）。读真实 SGLang 源码，讲 Attention Backends、CUDA Graph、量化这些进阶内容。

- <strong>Part IV：如何为 SGLang 做贡献</strong>（第 1 ~ 3 章）。把前面学到的东西变成一次真实的贡献。

&emsp;&emsp;推理优化是个靠动手的活，光看不练等于没学。代码一定要自己跑、自己改。遇到问题，直接来 issue 区问。

## 🤝 如何贡献

我们是一个开放的开源社区，欢迎任何形式的贡献：

- 🐛 <strong>报告 Bug</strong>：发现内容或代码问题，提交 Issue
- 💡 <strong>提出建议</strong>：对项目有好想法，欢迎发起讨论
- 📝 <strong>完善内容</strong>：帮助改进教程，提交 Pull Request
- ✍️ <strong>分享实践</strong>：分享你的学习笔记、benchmark 复现与实践经验

> 贡献之前，请先读一遍 [Part 0](docs/part0/Part0-编码伦理与开源精神.md)，了解我们期望的编码伦理与开源精神。

## 🔗 相关链接

- **仓库地址**：https://github.com/datawhalechina/zero-to-sglang
- **SGLang 官方**：https://github.com/sgl-project/sglang
- **mini-sglang 参考实现**：https://github.com/sgl-project/mini-sglang


## ❓ 常见问题

<details>
<summary><b>Q: 没有 GPU 可以学吗？</b></summary>

可以。Part I 完全不涉及 GPU；Part II 的大部分代码能在 CPU 上调试，但完整的性能验证和 Part III 的深入内容建议用 GPU（云服务也行）。
</details>

<details>
<summary><b>Q: 需要先精通 CUDA 吗？</b></summary>

不需要。这门课从推理的概念出发，Part III 才会深入 Attention Backends 等底层内容，到时候再按需补就行。
</details>

<!-- TODO: 可继续补充读者常见问题 -->

## 👥 贡献者

<!-- TODO: 补充贡献者名单（可参考 diy-llm / hello-agents 的表格形式） -->

*注：我们感谢每一位为项目做出贡献的开发者！*

### 特别感谢

- 感谢 Datawhle 和 SGLang 团队对项目的支持
- 感谢 [@Sm1les](https://github.com/Sm1les) 对本项目的帮助与支持
- 感谢所有为本项目做出贡献的开发者们 ❤️

## 🎓 引用

如果 zero-to-sglang 对您的研究或工作有所帮助，欢迎引用：

```bibtex
@misc{zero_to_sglang2026,
  title  = {zero-to-sglang: Building an LLM Inference Engine from Scratch},
  author = {TODO},
  year   = {2026},
  url    = {https://github.com/datawhalechina/zero-to-sglang},
  note   = {GitHub repository}
}
```

<!-- TODO: 补充 author 列表 -->

## ⭐ Star History

如果这个项目对你有帮助，欢迎给个 Star ⭐️！

<!-- TODO: 部署后可启用以下 Star History 图表 -->
<!--
<a href="https://www.star-history.com/?repos=datawhalechina%2Fzero-to-sglang&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=datawhalechina/zero-to-sglang&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=datawhalechina/zero-to-sglang&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=datawhalechina/zero-to-sglang&type=date&legend=top-left" />
 </picture>
</a>
-->

## 📄 许可证

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a>

本作品采用 [知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议](http://creativecommons.org/licenses/by-nc-sa/4.0/) 进行许可。

---

<div align="center">
  <p>让更多人能系统性地学会 LLM 推理引擎的构建</p>
  <p>Made with ❤️ by Datawhale & RadixArk</p>
</div>
