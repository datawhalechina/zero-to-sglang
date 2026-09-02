# 写作模板

怎么讲、讲多深、用什么口吻，都由作者自己定。这份模板只管两件事：**结构上的统一**，和**几条不能违反的硬性要求**。

## 文件与图片

- 一章一个 markdown 文件，放在 `docs/partN/` 下，文件名 `第N章_中文标题.md`。
- 图片放在同一 part 的 `docs/partN/images/` 下，文件名 `章号-序号-图片说明.png`，例如 `6-4-SM的架构.png`。
- 图片用 HTML 标签引用，宽度统一 800。

## 标题层级

一级标题全文只有一个，就是章节标题。格式为「第 N 章」加英文名，中文名跟在括号里：

```markdown
# 第 2 章 Introduction to Inference（推理入门）
```

二级标题是小节，三级标题编号为「小节号.序号」，四级标题为「小节号.序号.序号」：

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
