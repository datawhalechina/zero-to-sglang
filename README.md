<div align='center'>
    <img src="./course-material/eng/images/zero_to_sglang.png" alt="zero-to-sglang banner" width="100%">
    <h1>zero-to-sglang</h1>
    <p>
      <a href="./README.md"><img src="https://img.shields.io/badge/English-2563eb?style=for-the-badge" alt="English"></a>
      <a href="./README_zh.md"><img src="https://img.shields.io/badge/%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-6b7280?style=for-the-badge" alt="简体中文"></a>
    </p>
    <h3>📚 Build SGLang from Scratch</h3>
    <p><em>A hands-on tutorial on LLM inference: build a mini-sglang from scratch, then read the real SGLang source.</em></p>
</div>

<div align="center">
  <img src="https://img.shields.io/github/stars/datawhalechina/zero-to-sglang?style=flat&logo=github" alt="GitHub stars"/>
  <img src="https://img.shields.io/github/forks/datawhalechina/zero-to-sglang?style=flat&logo=github" alt="GitHub forks"/>
  <img src="https://img.shields.io/badge/language-English%20%7C%20Chinese-brightgreen?style=flat" alt="Language"/>
  <a href="https://github.com/datawhalechina/zero-to-sglang"><img src="https://img.shields.io/badge/GitHub-Project-blue?style=flat&logo=github" alt="GitHub Project"></a>
  <a href="https://datawhalechina.github.io/zero-to-sglang/eng/"><img src="https://img.shields.io/badge/Read-Online-green?style=flat&logo=gitbook" alt="Online Reading"></a>
</div>

<div align="center">
  <p><em>Start from what inference really is, build a mini-sglang by hand, then read the real SGLang source</em></p>
  <p>Datawhale × RadixArk</p>
</div>

---

> **Course status**: Part 0 and Part I are available in English, with further writing and review in progress. Parts II–IV are under development. You can switch to [简体中文](./README_zh.md) at any time. Want to help? See [Contributing](#-contributing).

## 🎯 About

&emsp;&emsp;Large models have moved on from the training race to the inference race. Once applications took off, inference cost and latency became the real bottleneck: on the same GPU, a good inference engine can serve several times more requests. Yet there are few tutorials on inference engines. They either stay at the concept level without touching code, or drop you straight into the SGLang source with no idea where to start.

&emsp;&emsp;This course fills that gap. The first half starts from the core questions of inference: why KV Cache exists, how prefill differs from decode, what compute-bound and memory-bound actually mean. The middle part walks you through building a mini-sglang from scratch: forward pass, generation, KV Cache, HTTP serving, Continuous Batching, Paged KV Cache, RadixAttention, added one at a time. The last part returns to the real SGLang, explains how its cutting-edge optimizations are actually implemented, and finally shows you how to land your first solid PR in SGLang.

&emsp;&emsp;The project is jointly launched by <strong>Datawhale</strong> and <strong>RadixArk</strong> (the company founded by the SGLang team). SGLang is technically solid and actively iterated in the inference framework space. If you like this project, please give the [official SGLang repository](https://github.com/sgl-project/sglang) a ⭐.

## ✨ What you will learn

- 📖 <strong>Understand inference</strong>: KV Cache, prefill/decode, compute-bound vs. memory-bound, and what these concepts actually mean
- 🏗️ <strong>Build the engine</strong>: write a mini-sglang from 0 to 1, with every step landing in code
- 🛠️ <strong>Read the real engine</strong>: how RadixAttention, Paged KV Cache and Continuous Batching are implemented inside SGLang
- ⚙️ <strong>Touch the frontier</strong>: quantization, hierarchical caching, DP Attention / EP / PP, Prefill-Decode disaggregation
- 🚀 <strong>Join open source</strong>: learn profiling and trace analysis, and walk through the full SGLang PR workflow

## 📋 Prerequisites

- **Python**: fluent in Python, with basic software engineering habits
- **Deep learning basics**: familiar with PyTorch and the fundamentals of neural networks
- **Math**: linear algebra and probability; knowing roughly how matrix multiplication and attention are computed is enough
- **GPU programming (optional)**: knowing CUDA basics helps, but is not required. Part I fills in the gaps from scratch
- **Hardware (optional)**: Part I needs no GPU; most of Part II can be debugged on CPU, but the full implementation and performance tests are best done on a GPU (cloud is fine)

## 🔗 Links

- **Repository**: https://github.com/datawhalechina/zero-to-sglang
- **SGLang**: https://github.com/sgl-project/sglang
- **mini-sglang reference implementation**: https://github.com/sgl-project/mini-sglang

## 📖 Syllabus

> Status legend: ✅ Done  🔄 In progress  📝 Needs polish  🚧 Planned  ⏸️ Paused
>
> Chapters in **bold** are written by SGLang core members.

| Chapter | Key content | Status |
|------|----------|------|
| <strong>Part 0 — Before you learn</strong> | | |
| [0.1 Coding Ethics and Open-Source Spirit](course-material/eng/part0/Part0-Coding-Ethics-and-Open-Source-Spirit.md) | Own your code, communicate like a human, profile first, the open-source spirit | ✅ |
| [0.2 Deploy Your First SGLang Server](course-material/eng/part0/Part0-Deploy-Your-First-SGLang-Server.md) | Environment setup; run Qwen3-0.6B with SGLang on your own GPU | ✅ |
| <strong>Part I — Foundations (concepts only, no code, no GPU)</strong> | | |
| [1. Introduction to LLM](course-material/eng/part1/Chapter1_Introduction_to_LLM.md) | What an LLM is and how it evolved, the Transformer architecture, autoregressive generation, key concepts | ✅ |
| [2. Introduction to Inference](<course-material/eng/part1/Chapter2_Introduction to Inference.md>) | Training vs. inference, prefill/decode, compute-bound vs. memory-bound, the Roofline model | ✅ |
| [3. Introduction to GPU](course-material/eng/part1/Chapter3_Introduction_to_GPU.md) | GPU architecture basics, how LLM inference executes on a GPU, understanding inference bottlenecks from the hardware | 🔄 |
| [4. KV Cache: The Core Data Structure of Inference](<course-material/eng/part1/Chapter4_KV Cache The Core Data Structure of Inference.md>) | Deriving KV Cache from attention, cache lifecycle, quantitative memory analysis | ✅ |
| [5. Introduction to Benchmark](course-material/eng/part1/Chapter5_Introduction_to_Benchmark.md) | Core metrics such as TTFT / TPOT / ITL / Goodput, percentiles and tail latency, how to design, run and read a benchmark | 🔄 |
| <strong>Part II — Build Your Own Mini SGL</strong> | | |
| 1. mini-sglang: what an inference engine looks like | Overall architecture of an inference engine, module breakdown, roadmap for this part | 📝 |
| 2. **Inside SGLang: The Path of a Request** | The full lifecycle of a request from arrival to response | 🚧 |
| 3. Your First 200 Lines: Forward Pass and Generation | Hand-writing the forward pass and the autoregressive generation loop | 🚧 |
| 4. KV Cache: From O(n²) to O(n) | Implementing the cache and optimizing attention computation | 🚧 |
| 5. Serving It: HTTP and Concurrent Requests | HTTP serving and handling concurrent requests | 🚧 |
| 6. Continuous Batching and the Scheduler | Continuous batching and scheduler design | 🚧 |
| 7. Paged KV Cache and Memory Management | Paged KV Cache and GPU memory management | 🚧 |
| 8. RadixAttention and Prefix Caching | RadixAttention and prefix caching | 🚧 |
| 9. Multi-process & Tensor Parallelism | Multi-process execution and tensor parallelism | 🚧 |
| 10. Speculative Decoding | Speculative decoding | 🚧 |
| <strong>Part III — Advanced Inference Technique (deep into the real SGLang)</strong> | | |
| 1. **Attention Backends (FlashInfer / Triton / FA3 / FlashMLA) and CUDA Graph** | Comparing mainstream attention backends, and CUDA Graph | 🚧 |
| 2. **Quantization and Low-Precision Inference** | Quantization and low-precision inference | 🚧 |
| 3. **Hierarchical Caching** | Hierarchical caching | 🚧 |
| 4. **Scaling Out: DP Attention, EP, PP** | Scaling out: DP Attention / EP / PP | 🚧 |
| 5. **Prefill-Decode Disaggregation** | Prefill-Decode disaggregation | 🚧 |
| <strong>Part IV — How to Make Contribution to SGLang (optional)</strong> | | |
| 1. **Deploying SGLang with the Cookbook** | Deploying SGLang with the Cookbook | 🚧 |
| 2. **Measuring It All: Profiling and Trace Analysis** | Profiling and trace analysis | 🚧 |
| 3. **SGLang PR Workflow** | The full SGLang PR workflow | 🚧 |

## 🗓️ Writing schedule

Start date: Monday, August 24, 2026. About one week per chapter; chapters in bold are written by SGLang core members.

| Part | Schedule | Status |
|------|----------|------|
| Part 0 — Before you learn | Done | ✅ |
| Part I — Foundations | Writing 8.24 ~ 9.06, review 9.07 ~ 9.13 | 🔄 |
| Part II — Build Your Own Mini SGL | Writing 9.14 ~ 10.04, review 10.05 ~ 10.10 | 🚧 |
| Part III — Advanced Inference Technique | 10.10 ~ 11.15 | 🚧 |
| Part IV — How to Make Contribution to SGLang | 11.16 ~ 12.6 | 🚧 |

## 🚀 Quick start

```bash
# Clone the repository
git clone https://github.com/datawhalechina/zero-to-sglang.git
cd zero-to-sglang
# Install dependencies as each chapter requires
```

### Learning path

1️⃣ Read Part I to build an overall picture of inference (no GPU needed)\
2️⃣ Follow Part II to build a mini-sglang from 0 to 1\
3️⃣ Move to Part III and understand the frontier optimizations against the real SGLang source\
4️⃣ Finish Part IV: profiling, trace analysis, and the full PR workflow

### Project structure

```
zero-to-sglang/
├── course-material/         # Course text
│   ├── ch/                  # Chinese edition (read online at /ch/)
│   │   ├── part0/           # Part 0: Before you learn
│   │   ├── part1/           # Part I: Foundations (in progress)
│   │   ├── part2/           # Part II: Build Your Own Mini SGL (planned)
│   │   ├── part3/           # Part III: Advanced Inference Technique (planned)
│   │   ├── part4/           # Part IV: How to Make Contribution to SGLang (planned)
│   │   └── WRITING_TEMPLATE.md
│   └── eng/                 # English edition (read online at /eng/, in progress)
│       ├── part0/
│       └── WRITING_TEMPLATE.md
├── community/               # Community corner: build logs, pitfalls, study notes
│   ├── ch/
│   └── eng/
├── docs/.vitepress/         # VitePress site config
├── README.md                # Project overview (English, default)
├── README_zh.md             # Project overview (Chinese)
├── README_en.md             # Legacy English entry (links to README.md)
└── .gitignore               # Git ignore rules
```

## 🤝 Contributing

We are an open community and welcome contributions of every kind. Before you start, please read [Part 0](course-material/eng/part0/Part0-Coding-Ethics-and-Open-Source-Spirit.md) to understand the coding ethics and open-source spirit we expect.

<strong>1. Improve the course text</strong>

- 🐛 Found an error in the content, a formula, or code: open an issue, or fix it and open a PR
- 📝 Add to a chapter or improve the wording: open a PR. Chapter division and numbering follow the outline in the [writing template](course-material/eng/WRITING_TEMPLATE.md); do not add, remove, or merge chapters on your own
- 🌐 Translate: help translate chapters into English following the [English writing template](course-material/eng/WRITING_TEMPLATE.md), and put them under `course-material/eng/`
- 💡 Have ideas about the course: open an issue to discuss

<strong>2. Share your practice</strong>

- 🔧 Built mini-sglang yourself following Part II? Write down the bugs you hit and how you fixed them
- ⚠️ Pitfalls from deploying SGLang or setting up the environment, or places where the docs were unclear and others deserve a warning
- 📒 Study notes written in your own words after finishing a chapter, or extra derivations and experiments

Community content goes under the matching directory in `community/eng/` (English) or `community/ch/` (Chinese).

<strong>How to open a PR</strong>

1. Fork the repository and create a branch from `main`
2. PR requirements are in [community/eng/PR_requirement.md](community/eng/PR_requirement.md); check every item before submitting

## ❓ FAQ

<details>
<summary><b>Q: Can I follow the course without a GPU?</b></summary>

Yes. Part I involves no GPU at all. Most of the code in Part II can be debugged on CPU, but full performance verification and the deeper content of Part III are best done on a GPU (cloud is fine).
</details>

<details>
<summary><b>Q: Do I need to master CUDA first?</b></summary>

No. The course starts from inference concepts. Low-level topics such as attention backends only come in Part III, and you can catch up on CUDA as needed at that point.
</details>


## 👥 Contributors


*We thank every developer who has contributed to this project!*


### Special thanks

- Thanks to Datawhale and the SGLang team for supporting the project
  - Datawhale: [@1iyouzhen](https://github.com/1iyouzhen), [@xuhu0115](https://github.com/xuhu0115), [@kangkang-Adam](https://github.com/kangkang-Adam)
  - SGLang: [@Ccyest](https://github.com/Ccyest), [@fcranzhou](https://github.com/fcranzhou)
- Thanks to [@Sm1les](https://github.com/Sm1les) for the help and support
- Thanks to every developer who has contributed to this project ❤️

## 🎓 Citation

If zero-to-sglang helps your research or work, please cite it:

```bibtex
@misc{zero_to_sglang2026,
  title  = {zero-to-sglang: Building an LLM Inference Engine from Scratch},
  author = {TODO},
  year   = {2026},
  url    = {https://github.com/datawhalechina/zero-to-sglang},
  note   = {GitHub repository}
}
```

<!-- TODO: fill in the author list -->

## ⭐ Star History

If this project helps you, please give it a Star ⭐️!

<!-- TODO: enable the Star History chart after deployment -->
<!--
<a href="https://www.star-history.com/?repos=datawhalechina%2Fzero-to-sglang&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=datawhalechina/zero-to-sglang&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=datawhalechina/zero-to-sglang&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=datawhalechina/zero-to-sglang&type=date&legend=top-left" />
 </picture>
</a>
-->

## 📄 License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a>

This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

---

<div align="center">
  <p>Helping more people learn to build LLM inference engines, systematically</p>
  <p>Made with ❤️ by Datawhale & RadixArk</p>
</div>
