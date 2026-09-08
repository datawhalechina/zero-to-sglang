# Chapter 5 — Introduction to Benchmark

The previous chapters analyzed the inference process and distinguished the compute and memory-access characteristics of Prefill and Decode. These conclusions are still qualitative. To verify system optimizations, compare different engines, or plan deployment capacity, we need quantitative performance data. A benchmark provides such data while keeping results reproducible and comparable.

## 1. Learning objectives

The rest of this course will build a mini-sglang from scratch and analyze SGLang in depth. Every change needs to be validated quantitatively with benchmarks. By the end of this chapter, you should be able to:

- distinguish latency and throughput, and understand TTFT, TPOT, ITL, throughput, and Goodput;
- understand percentiles and tail latency;
- choose a suitable benchmark workload and design a reproducible, comparable test;
- run SGLang benchmark tools and interpret their results.

## 2. What is a benchmark and why does it matter?

### 2.1 Introduction

When you open an evaluation website or paper about a large language model (LLM), what do you usually see first?

<div align="center">
   <img src="./images/5-1-deepseek-r1-benchmark-performance.png" width="800"/>
   <p>Figure 5.1 Benchmark performance of DeepSeek-R1</p>
 </div>

This is the most direct and common form of evaluation. Major models usually report scores on standardized benchmarks.

Evaluation should not focus on capability alone. Cost and inference speed are also important dimensions. For example, [Artificial Analysis](https://artificialanalysis.ai/) evaluates models from the perspectives of intelligence, inference speed, and price:

<div align="center">
   <img src="./images/5-2-frontier-model-performance-on-artificial-analysis.png" width="800"/>
   <p>Figure 5.2 Model performance rankings on Artificial Analysis</p>
 </div>

Websites such as Artificial Analysis present model capability, inference speed, and cost per token together, helping readers understand the performance-cost trade-off. In this chapter, the example only illustrates how evaluation results can be presented; it does not judge any specific model.

<div align="center">
   <img src="./images/5-3-frontier-model-pareto-performance-on-artificial-analysis.png" width="800"/>
   <p>Figure 5.3 Performance-versus-cost comparison on Artificial Analysis</p>
 </div>

The central question of **evaluation** is: given a fixed model, how good is it? To understand this chapter, distinguish two kinds of benchmarks:

- **Model Capability Benchmark**: evaluates the model itself. MMLU, GSM8K, HumanEval, and SWE-bench measure output quality in knowledge understanding, mathematical reasoning, code generation, and other capabilities. It asks whether the model gives a good answer.
- **Inference Workload Benchmark**: evaluates a deployed inference service under a request workload. It measures TTFT, TPOT, throughput, P99 latency, and other engineering metrics. It asks whether the model runs quickly and reliably, and is also commonly called an AI Infra benchmark.

These benchmarks differ in evaluation target, input, and metrics. The first gives tasks to a model and checks answer quality; the second sends requests to an inference service and observes system performance. They must not be conflated.

> Model capability benchmarks are not the focus of this chapter. Unless stated otherwise, “benchmark” means an `LLM inference workload benchmark`. For more on model evaluation, see [Evaluation and Benchmarking](https://github.com/datawhalechina/diy-llm/blob/main/docs/zh/chapter12/chapter12_%E8%AF%84%E4%BC%B0%E4%B8%8E%E5%9F%BA%E5%87%86%E6%B5%8B%E8%AF%95.md).

### 2.2 Why use benchmarks?

A benchmark is a reproducible and comparable test. It helps us understand inference-service and hardware bottlenecks, quantify optimization effects, and plan production capacity and resource allocation.

## 3. Core metrics

Inference-service metrics fall into two categories: **latency**, which describes the response speed of an individual request, and **throughput**, which describes the system's overall output capacity. They answer different questions.

### 3.1 Latency metrics

There are four latency metrics:

<div align="center">
  <img src="./images/5-4-request-timeline.png" alt="5-4-request-timeline.png" width="800">
  <p><em>Figure 5.4 Timeline of a request</em></p>
</div>

- **TTFT (Time to First Token)**: the time from sending a request to generating the first token, including queueing, Prefill, and the first Decode step.
- **TPOT (Time Per Output Token)**: the average generation time per token after the first token, mainly determined by Decode.
- **ITL (Inter-Token Latency)**: the interval between adjacent tokens. TPOT is an average, while ITL is observed token by token and can reveal jitter.
- **E2E latency**: the total time from sending a request until it completes.

Suppose a request has an input of 1,000 tokens, generates 200 tokens, TTFT is 250 ms, and TPOT is 25 ms. The remaining 199 tokens take:

$$\text{Remaining generation time} = 25 \times 199 = 4975 \text{ ms}$$

$$\text{E2E} = \text{TTFT} + \text{Remaining generation time} = 250 + 4975 = 5225 \text{ ms} \approx 5.2 \text{ s}$$

Prefill accounts for only 250 ms; almost all remaining time comes from Decode. Therefore, optimizing TPOT is more beneficial for **long-output** workloads, while TTFT is the main concern for **short-output, long-input** workloads such as document question answering.

Converting TPOT into a generation rate is intuitive: 25 ms/token corresponds to 40 tokens/s. A common reference lower bound is 100 ms/token, or 10 tokens/s.

### 3.2 Throughput metrics

- **Throughput**: output per unit of time, measured in `token/s` or `request/s`.
- **Goodput**: throughput subject to latency constraints. If TTFT must stay below 500 ms, requests beyond that limit do not contribute to Goodput.

Higher concurrency can increase throughput but also increases latency, so Goodput may be more useful than raw Throughput. Higher concurrency and larger batches improve GPU utilization, but can cause longer queueing and worse TTFT.

<div align="center">
  <img src="./images/5-5-latency-throughput.jpg" alt="5-5-latency-throughput.jpg" width="800">
  <p><em>Figure 5.5 Throughput and TTFT as concurrency changes (constructed example)</em></p>
</div>

In Figure 5.5, the blue curve is throughput, the orange curve is TTFT, and the horizontal axis is concurrency. The data are constructed examples. Throughput rises quickly and then approaches saturation, while TTFT increases as concurrency grows. Higher concurrency is therefore not always better.

| Scenario | Key metrics |
|------|---------|
| Dialogue / chat | TTFT |
| Long-form generation / writing | TPOT |
| Offline batch processing / evaluation | Throughput |
| Real-time speech / multi-step Agent reasoning | TTFT and tail latency |

## 4. How to design a benchmark

The first question is what you want to measure. Only then should you choose the dataset or request set and fix runtime parameters.

### 4.1 Define the evaluation target and workload

This section focuses on **inference workload benchmarks**, not model capability benchmarks. Model capability benchmarks use MMLU, GSM8K, HumanEval, and SWE-bench to check output quality. Inference workload benchmarks use real or constructed request sets to apply load to an inference service and measure latency, throughput, and resource efficiency. The former asks “How capable is the model?”; the latter asks “How well does the inference system perform?”

The workload must match the evaluation target. Model capability evaluation can use tasks for knowledge, mathematical reasoning, or code generation. Inference-service evaluation should use request distributions matching the target application, such as real conversations, fixed-length requests, shared-prefix requests, or multi-turn traces with tool calls.

The following figure summarizes representative categories of LLM benchmarks:

<div align="center">
  <img src="./images/5-6-llm-benchmarks-taxonomy.png" alt="5-6-llm-benchmarks-taxonomy.png" width="800">
  <p><em>Figure 5.6 Representative categories of LLM benchmarks</em></p>
</div>

SGLang provides four benchmark tools at different engine layers:

<div align="center">
  <img src="./images/5-7-benchmark-tools.png" alt="5-7-benchmark-tools.png" width="800">
  <p><em>Figure 5.7 Engine layers and benchmark-tool coverage</em></p>
</div>

| Tool | Through HTTP | Through scheduler | Purpose |
|------|--------|---------|------|
| `bench_serving` | Yes | Yes | Closest to an online workload; reports TTFT / TPOT / ITL / throughput and percentiles |
| `bench_one_batch_server` | Yes | Yes | End-to-end latency for one batch, including HTTP and scheduling overhead |
| `bench_offline_throughput` | No | Yes | Measures maximum throughput without network overhead |
| `bench_one_batch` | No | No | Calls the kernel directly for kernel-level analysis |

In most cases, `bench_serving` is sufficient:

- **sharegpt**: real conversation data, close to chat workloads;
- **random**: fixed-length random requests for testing the performance ceiling, not representative of real traffic;
- **generated-shared-prefix**: requests with a shared prefix, used to measure prefix reuse;
- **agentic-trace**: multi-turn traces with tool calls, corresponding to Agent workloads.

### 4.2 Fix runtime parameters

- Fix concurrency or request rate with `--max-concurrency` or `--request-rate`.
- Fix hardware, model, precision, and quantization method. FP16 and INT8 can differ by more than 2x in throughput.
- Warm up sufficiently so that CUDA Graphs and the KV-cache allocator reach a steady state.
- Fix output length or its distribution; otherwise throughput numbers are not comparable.

## 5. How to interpret benchmark results

### 5.1 Look at latency before throughput

First check whether latency meets the requirement, and then examine throughput. Define the latency target, or SLA (Service Level Agreement), for TTFT and TPOT. Examine the distribution rather than the mean alone; P50 and P99 should be observed together.

The mean cannot describe the tail of a latency distribution. The figure below illustrates TTFT measurements:

<div align="center">
  <img src="./images/5-8-percentiles.png" alt="5-8-percentiles.png" width="800">
  <p><em>Figure 5.8 TTFT distribution and percentiles</em></p>
</div>

P50 is approximately 202 ms, meaning that half of requests are faster than this value. The mean is pulled up to 246 ms by the long tail, while P99 reaches 862 ms. Latency should therefore be evaluated with percentiles, especially P90 and P99. SGLang benchmark tools output mean, median, P90, P95, and P99 by default.

When comparing results, check the model, precision, hardware, concurrency, request set, and output length. If any condition differs, the comparison may no longer be meaningful.

### 5.2 Common pitfalls

1. Looking at throughput without latency. Throughput at concurrency 64 may be higher than at concurrency 16, while P99 TTFT may deteriorate from 300 ms to 3 s.
2. Looking at the mean without tail latency. One very slow request can affect users even when the average looks acceptable. Track P50, P95, and P99 together.
3. Measuring a cold-start state because the service was not warmed up or too few requests were sent. Perform a warm-up phase before formal measurement.
4. Treating `bench_one_batch` as online performance. It bypasses the scheduler and HTTP layer and is intended for kernel-level analysis only.
5. Comparing throughput with workloads of different lengths. Input and output lengths affect Prefill and Decode computation. Use the same request set and similar length distributions when comparing optimizations.
6. Ignoring tokenizer differences when comparing token/s across models. The same text can produce different token counts with different tokenizers, so token/s cannot be interpreted simply as model speed.
7. Looking only at token/s and ignoring Goodput. Requests that violate the latency SLA contribute to throughput but may have no business value.
8. Drawing conclusions from a single-machine measurement while ignoring communication overhead and topology in multi-GPU or multi-machine deployments. Tensor parallelism needs NVLink or RDMA, and multi-machine deployments also incur network latency.
9. Failing to fix the random seed. Sampling kernels can take different execution paths, causing noise that makes optimization effects difficult to identify.

## 6. Summary and exercises

### 6.1 Summary

This chapter introduced the role of benchmarks, core metrics, benchmark design, and result interpretation. Distinguish latency (TTFT / TPOT / ITL / E2E) from throughput (Throughput / Goodput), and evaluate latency with percentiles rather than the mean alone. The value of a benchmark lies in comparability: define the latency target, inspect the distribution, and check every experimental condition. This methodology will later be used to evaluate mini-sglang and SGLang.

### 6.2 Exercises

1. What do TTFT, TPOT, and ITL measure? Why is optimizing TPOT more beneficial than optimizing TTFT for long-output workloads?

   > Hint: Consider the roles of Prefill and Decode.

2. What does it mean when the mean and P99 differ by an order of magnitude? Why should service stability focus on P99?

3. In a chat workload, users see no response for a long time, but generation is fast once it begins. Which metric is abnormal, and what should be optimized?

4. Why is a direct token/s comparison across models unreliable?

   > Hint: Consider tokenizers.

5. Which benchmark should be used for a language model focused on mathematical reasoning and for a speech-synthesis model? Why can they not share the same dataset?

6. List the considerations required for a reproducible and comparable benchmark.

   > Hint: Cover concurrency, hardware, warm-up, output length, and reported metrics.

## References

- [SGLang Documentation](https://docs.sglang.io/docs/developer_guide/benchmark_and_profiling#benchmark)
- [Chapter 12 of diy-llm: Evaluation and Benchmarking](https://github.com/datawhalechina/diy-llm/blob/main/docs/zh/chapter12/chapter12_%E8%AF%84%E4%BC%B0%E4%B8%8E%E5%9F%BA%E5%87%86%E6%B5%8B%E8%AF%95.md)
- [Response Times: The 3 Important Limits](https://www.nngroup.com/articles/response-times-3-important-limits/)
- [GSM8K: Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)
- [Measuring Massive Multitask Language Understanding (MMLU)](https://arxiv.org/abs/2009.03300)
- [NVIDIA NIM LLMs Benchmarking](https://docs.nvidia.com/nim/benchmarking/llm/latest/overview.html)
- [A Brief Introduction to LLM Inference Benchmarking](https://rudeigerc.dev/posts/llm-inference-benchmarking/)
- [A Survey on Large Language Model Benchmarks](https://arxiv.org/abs/2508.15361v1)
- [DeepSeek-R1: Incentivizing Reasoning Capability in Large Language Models via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
