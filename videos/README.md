# 课程配套视频 · Companion videos

每节课一个 3 分钟左右的讲解视频，中英文各一版，按 Part 和章节放在对应文件夹里。
One ~3-minute explainer per lesson, in Chinese and English, grouped by part and chapter.

## Part 0 · 开课之前 / Before you learn

| 章节 · Chapter | 中文 | English |
|---|---|---|
| 编码伦理与开源精神 · Coding Ethics and the Open-Source Spirit | [▶ 3:07](part0/0.1-coding-ethics-zh.mp4) | [▶ 3:27](part0/0.1-coding-ethics-en.mp4) |
| 部署你的第一个 SGLang 服务 · Deploying Your First SGLang Server | [▶ 3:17](part0/0.2-deploy-first-server-zh.mp4) | [▶ 3:36](part0/0.2-deploy-first-server-en.mp4) |

## Part I · 基础 / Foundations

| 章节 · Chapter | 中文 | English |
|---|---|---|
| LLM 入门：预测下一个 token · Intro to LLMs: Predicting the Next Token | [▶ 3:35](part1/ch1-llm-intro-zh.mp4) | [▶ 4:04](part1/ch1-llm-intro-en.mp4) |
| 推理入门：Prefill 与 Decode · Intro to Inference: Prefill and Decode | [▶ 3:21](part1/ch2-inference-intro-zh.mp4) | [▶ 3:43](part1/ch2-inference-intro-en.mp4) |
| GPU 入门：为吞吐而生 · GPU Basics: Built for Throughput | [▶ 3:28](part1/ch3-gpu-intro-zh.mp4) | [▶ 3:44](part1/ch3-gpu-intro-en.mp4) |
| KV Cache：推理的核心数据结构 · KV Cache: The Core Data Structure of Inference | [▶ 2:49](part1/ch4-kv-cache-zh.mp4) | [▶ 3:05](part1/ch4-kv-cache-en.mp4) |
| Benchmark 入门 · Intro to Benchmarking | [▶ 3:43](part1/ch5-benchmark-zh.mp4) | [▶ 4:03](part1/ch5-benchmark-en.mp4) |

## Part II · 从零手写 mini-sglang / Build Your Own Mini SGL

| 章节 · Chapter | 中文 | English |
|---|---|---|
| mini-sglang：推理引擎长什么样 · mini-sglang: What an Inference Engine Looks Like | [▶ 3:03](part2/ch1-mini-sglang-zh.mp4) | [▶ 3:20](part2/ch1-mini-sglang-en.mp4) |
| 一个请求的旅程 · The Journey of a Request | [▶ 3:38](part2/ch2-request-path-zh.mp4) | [▶ 3:57](part2/ch2-request-path-en.mp4) |
| 前向与生成 · Forward Pass & Generation | [▶ 3:31](part2/ch3-forward-and-generation-zh.mp4) | [▶ 3:42](part2/ch3-forward-and-generation-en.mp4) |
| KV Cache 优化 · KV Cache Optimization | [▶ 3:41](part2/ch4-kv-cache-optimization-zh.mp4) | [▶ 4:06](part2/ch4-kv-cache-optimization-en.mp4) |
| HTTP 服务与并发 · HTTP Serving and Concurrency | [▶ 3:29](part2/ch5-http-and-concurrency-zh.mp4) | [▶ 3:47](part2/ch5-http-and-concurrency-en.mp4) |
| Continuous Batching 与调度 · Continuous Batching and Scheduling | [▶ 3:54](part2/ch6-continuous-batching-zh.mp4) | [▶ 4:18](part2/ch6-continuous-batching-en.mp4) |
| Paged KV Cache 与显存管理 · Paged KV Cache and Memory Management | [▶ 3:28](part2/ch7-paged-kv-cache-zh.mp4) | [▶ 3:35](part2/ch7-paged-kv-cache-en.mp4) |
| RadixAttention 与前缀缓存 · RadixAttention and Prefix Caching | [▶ 3:55](part2/ch8-radix-attention-zh.mp4) | [▶ 4:12](part2/ch8-radix-attention-en.mp4) |
| 多进程与张量并行 · Multi-Process and Tensor Parallelism | [▶ 4:00](part2/ch9-tensor-parallelism-zh.mp4) | [▶ 4:21](part2/ch9-tensor-parallelism-en.mp4) |
| 投机解码：先猜几个，再一次验证 · Speculative Decoding: Guess a Few, Verify at Once | [▶ 4:02](part2/ch10-speculative-decoding-zh.mp4) | [▶ 4:27](part2/ch10-speculative-decoding-en.mp4) |

## Part III · 真实 SGLang / Real SGLang

| 章节 · Chapter | 中文 | English |
|---|---|---|
| 注意力后端：同一个公式，四种实现 · Attention Backends: One Formula, Four Implementations | [▶ 4:00](part3/ch1-attention-backends-zh.mp4) | [▶ 4:44](part3/ch1-attention-backends-en.mp4) |
| 分层缓存：GPU 之外的前缀缓存 · Hierarchical Caching: Prefix Cache Beyond the GPU | [▶ 4:10](part3/ch3-hierarchical-cache-zh.mp4) | [▶ 4:22](part3/ch3-hierarchical-cache-en.mp4) |

## 制作 · Credits

- 配音：开源语音模型 [MOSS-TTS](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5)（OpenMOSS），克隆作者本人声音；配乐：MiniMax-Music3；均由 [SGLang-Omni](https://github.com/sgl-project/sglang-omni) 推理。
- Narration: the open MOSS-TTS model (OpenMOSS), cloned from the author's own voice; music: MiniMax-Music3; both served with SGLang-Omni.
- 画面用 three.js 逐帧渲染 · Visuals rendered frame by frame with three.js.
