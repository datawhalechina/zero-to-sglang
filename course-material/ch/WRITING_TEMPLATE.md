# 写作模板

怎么讲、讲多深、用什么口吻，都由作者自己定。这份模板只管两件事：**结构上的统一**，和**几条不能违反的硬性要求**。

CI 会检查章节的引言位置、段首缩进、标题层级、结尾结构和图片格式；引言是否讲清背景、学习目标是否准确，仍需作者和 reviewer 通读确认。每个 PR 都会跑 `python3 scripts/check_chapters.py --all`。提交前也请在本地运行这条命令；只检查单个文件可运行 `python3 scripts/check_chapters.py course-material/ch/part1/第2章_推理入门.md`。

## 章节开头与段落格式

以[第 1 章 LLM 入门](./part1/第1章_LLM入门.md)为排版基准，中英文遵循同一套结构。

- 开头固定为「一级章节标题 → 引言 → `## 1 本章学习目标` → 正文小节」。
- 引言紧接章节标题，放在第一个二级标题之前，用一至数段正文说明背景、本章主题及其与前后章节的联系。不要另设「引言」标题，也不要用学习目标列表代替引言。
- 普通正文段落从行首开始，**禁止段首缩进**：不加空格、Tab、全角空格或 Unicode 宽空格，不用 `&emsp;`、`&ensp;`、`&nbsp;` 及其数字实体，也不用 HTML/CSS 制造首行缩进。
- 段落之间、标题与内容之间空一行。不要用连续空行、`<br>` 或行尾空格模拟段落间距。
- 代码块中的缩进、嵌套列表及其续行的缩进、引用标记后的一个分隔空格、公式与 HTML 容器内部的结构缩进应保留；这些不是正文首行缩进。代码必须使用围栏代码块。

Part 0 的两篇导读也必须有标题后的引言，并遵守段落格式；一级标题分别为 `# Part 0.1 编码伦理与开源精神` 和 `# Part 0.2 部署你的第一个 SGLang 服务`，正文保留导读和操作指南的小节结构，不强制套用 Part I–IV 的学习目标、编号与结尾。只有「标题 + 正在编写提示」的占位章节暂不要求引言、学习目标与结尾；开始写正文时必须补齐。

## 可复制的章节骨架

复制后替换章节号、标题和占位说明。新增正文小节时，连续编号，并同步调整总结小节编号。

```markdown
# 第 N 章 章节标题

这里写引言：说明背景、本章主题，以及它与前后章节的联系。

## 1 本章学习目标

完成本章学习后，你将能够：

1. 学习目标一。
2. 学习目标二。

## 2 正文小节标题

正文从行首开始，段首不加任何空格或空白实体。

### 2.1 子节标题

下一段与上一段之间空一行。

## 3 总结与测试题

### 3.1 课程总结

概括本章的关键知识。

### 3.2 测试题

1. 这里写测试题，只出题，不给答案。

## 参考资料

- [实际引用的资料标题](https://example.com/source)
```

## 文件与图片

- 一章一个 markdown 文件，放在 `course-material/ch/partN/` 下，文件名 `第N章_中文标题.md`。
- 图片放在同一 part 的 `course-material/ch/partN/images/` 下，文件名 `章号-序号-图片说明.png`，例如 `6-4-SM的架构.png`。
- 图片用 HTML 标签引用，宽度统一 800。

## 标题层级

一级标题全文只有一个，就是章节标题。格式为「第 N 章」加中文名；英文名不用写，英文版有单独的文档：

```markdown
# 第 2 章 推理入门
```

二级标题是小节，从 `## 1 本章学习目标` 开始；三级标题编号为「小节号.序号」，四级标题为「小节号.序号.序号」：

```markdown
## 2 balabala

### 2.1 balabala

#### 2.1.1 balabala
```

四级以下不再往下分，还需要分层的地方用无序列表或加粗行。同一层级内编号必须连续，不跳号、不重号。不要用一级标题充当小节标题。

## 结尾

每章固定用两个二级标题收尾。

第一个是本章最后一节，标题为「总结与测试题」，下面带两个三级标题：

```markdown
## 6 总结与测试题

### 6.1 课程总结

### 6.2 测试题
```

测试题只出题，不给答案。

第二个是参考资料，不参与小节编号：

```markdown
## 参考资料
```

无序列表，一行一条，写清标题和链接，只列真正引用过的资料。

## 不能违反的两条

### 1 不点名其他项目

这是 SGLang 官方课程。正文、图表、代码注释、参考资料里都不要指名道姓提到其他推理项目（比如 vLLM），更不要拿它们的数字来做优劣对比——很容易引起商业纠纷。

需要对比时，讲技术方案本身，不挂项目名；实在要提，用「其他推理引擎」「主流实现」这类中性说法。

### 2 不改动大纲

章节的划分、编号和顺序按下面的大纲来，不要自行增删、合并或调换。确实需要调整，先改这份大纲和 README 的章节表，再动文件。

## 课程大纲

**Part 0 — Before you learn**

1. Coding ethics and open-source spirit
2. Environment setup

**Part I — Foundations**（concepts only，无代码、无 GPU）

1. Introduction to LLM
2. Introduction to inference
3. Introduction to GPU
4. KV Cache: The Core Data Structure of Inference
5. Introduction to Benchmark

**Part II — Build Your Own Mini SGL**（从 0 到 1 手搓 mini-sglang）

1. mini-sglang, what an inference engine looks like
2. Inside SGLang: The Path of a Request
3. Your First 200 Lines: Forward Pass and Generation
4. KV Cache: From O(n²) to O(n)
5. Serving It: HTTP and Concurrent Requests
6. Continuous Batching and the Scheduler
7. Paged KV Cache and Memory Management
8. RadixAttention and Prefix Caching
9. Multi-process & Tensor Parallelism
10. Speculative Decoding

**Part III — Advanced Inference Technique**（深入真实 SGLang）

1. Attention Backends（FlashInfer / Triton / FA3 / FlashMLA）and CUDA Graph
2. Quantization and Low-Precision Inference
3. Hierarchical Caching
4. Scaling Out: DP Attention, EP, PP
5. Prefill-Decode Disaggregation

**Part IV — How to Make Contribution to SGLang**（可选，可按需删减）

1. Deploying SGLang with the Cookbook
2. Measuring It All: Profiling and Trace Analysis
3. SGLang PR Workflow
