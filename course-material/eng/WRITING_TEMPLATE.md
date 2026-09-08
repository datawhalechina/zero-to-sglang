# Writing Template

How to explain, how deep to go, and what tone to use are up to the author. This template only covers two things: **a uniform structure**, and **a few hard rules that must not be broken**.

The rules below are enforced by CI: every chapter file changed in a PR is run through `python3 scripts/check_chapters.py`, and a PR that fails cannot be merged. Run it locally before submitting: `python3 scripts/check_chapters.py course-material/eng/part1/Chapter2_Introduction-to-Inference.md`, or `--all` for every chapter.

## Files and images

- One chapter per markdown file, under `course-material/eng/partN/`, named `ChapterN_English-Title.md`, e.g. `Chapter2_Introduction-to-Inference.md`.
- Images go in the same part's `course-material/eng/partN/images/`, named `chapter-index-description.png`, e.g. `6-4-SM-architecture.png`. If a figure has no text that needs translating, reference the Chinese edition's copy by relative path instead of duplicating it.
- Reference images with an HTML tag, width 800.

## Heading levels

There is exactly one level-1 heading in the whole file: the chapter title, in the form "Chapter N" followed by the chapter name:

```markdown
# Chapter 2 Introduction to Inference
```

Level-2 headings are sections. Level-3 headings are numbered "section.index", level-4 headings "section.index.index":

```markdown
## 2 balabala

### 2.1 balabala

#### 2.1.1 balabala
```

Do not go below level 4; where you still need to subdivide, use bullet lists or bold lines. Numbering within a level must be continuous: no skipped or duplicated numbers. Do not use level-1 headings as section headings.

## Ending

Every chapter ends with two fixed level-2 headings.

The first is the chapter's last section, titled "Summary and Exercises", with two level-3 headings under it:

```markdown
## 6 Summary and Exercises

### 6.1 Summary

### 6.2 Exercises
```

Exercises pose questions only, with no answers.

The second is the references section, which is not part of the section numbering:

```markdown
## References
```

A bullet list, one entry per line, with title and link. List only material you actually cited.

## Staying in sync with the Chinese edition

English chapters are translations of the Chinese originals in `course-material/ch/`. Keep the section structure and numbering identical to the Chinese chapter so readers can switch editions on any page. Translate the meaning, not the words: rewrite sentences so they read naturally in English, but do not add, drop, or reorder content. If the Chinese chapter changes later, update the English chapter in the same PR, or open an issue so someone can.

## Two rules that must not be broken

### 1 Do not name other projects

This is the official SGLang course. In the body text, figures, code comments, and references, do not name other inference projects (such as vLLM), and never quote their numbers in a better-or-worse comparison. That invites commercial disputes.

When a comparison is needed, discuss the technical approach itself without attaching a project name. If you really must refer to one, use neutral wording such as "other inference engines" or "mainstream implementations".

### 2 Do not change the outline

Chapter division, numbering, and order follow the outline below. Do not add, remove, merge, or reorder chapters on your own. If a change is really needed, update this outline and the chapter table in the README first, then touch the files.

## Course outline

**Part 0 — Before you learn**

1. Coding ethics and open-source spirit
2. Environment setup

**Part I — Foundations** (concepts only, no code, no GPU)

1. Introduction to LLM
2. Introduction to inference
3. Introduction to GPU
4. KV Cache: The Core Data Structure of Inference
5. Introduction to Benchmark

**Part II — Build Your Own Mini SGL** (build mini-sglang from 0 to 1)

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

**Part III — Advanced Inference Technique** (deep into the real SGLang)

1. Attention Backends (FlashInfer / Triton / FA3 / FlashMLA) and CUDA Graph
2. Quantization and Low-Precision Inference
3. Hierarchical Caching
4. Scaling Out: DP Attention, EP, PP
5. Prefill-Decode Disaggregation

**Part IV — How to Make Contribution to SGLang** (optional, trim as needed)

1. Deploying SGLang with the Cookbook
2. Measuring It All: Profiling and Trace Analysis
3. SGLang PR Workflow
