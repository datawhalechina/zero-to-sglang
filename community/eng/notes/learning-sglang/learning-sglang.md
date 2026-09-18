# Learning SGLang: an illustrated, step-by-step tour of an inference engine

**Site:** https://wilsonzheng0327.github.io/learning-sglang/

**Source:** https://github.com/WilsonZheng0327/learning-sglang

<img src="./images/learning-sglang.gif" alt="Stepping through a chapter of Learning SGLang" width="800">

While working through the course I built an illustrated companion to it: a small site where each chapter is one animated visualization you click through. The captions are the narration; there is almost no other text. Every chapter ends on a question, and the next chapter is the answer, so by the end an engine's scheduler, memory manager, prefix cache, and process layout all read as the obvious response to a chain of "so what do we do about that?"

Roughly, chapters 1 to 4 cover the ground of the course's Part I and chapters 5 to 10 the ground of Part II. They are meant to be read alongside a chapter, not instead of it.

## Chapters

| # | Chapter | Hook |
|---|---|---|
| 01 | [Bare-minimum inference](https://wilsonzheng0327.github.io/learning-sglang/ch/01-inference/) | One function, called in a loop. |
| 02 | [Attention, per decode step](https://wilsonzheng0327.github.io/learning-sglang/ch/02-attention/) | What a new token needs from the past. |
| 03 | [KV cache](https://wilsonzheng0327.github.io/learning-sglang/ch/03-kv-cache/) | Keep k and v. Drop q. |
| 04 | [Prefill vs decode](https://wilsonzheng0327.github.io/learning-sglang/ch/04-prefill-decode/) | Two very different workloads. |
| 05 | [Batching & continuous batching](https://wilsonzheng0327.github.io/learning-sglang/ch/05-batching/) | Sharing a GPU between users. |
| 06 | [The scheduler](https://wilsonzheng0327.github.io/learning-sglang/ch/06-scheduler/) | Two lists, one GPU, one choice per step. |
| 07 | [KV memory](https://wilsonzheng0327.github.io/learning-sglang/ch/07-kv-memory/) | Pages, and the memory cap. |
| 08 | [Prefix caching](https://wilsonzheng0327.github.io/learning-sglang/ch/08-prefix-caching/) | Same beginning, one copy. |
| 09 | [One request, end to end](https://wilsonzheng0327.github.io/learning-sglang/ch/09-one-request/) | From an HTTP POST to the GPU and back. |
| 10 | [The scheduler's loop](https://wilsonzheng0327.github.io/learning-sglang/ch/10-engine/) | One Req through one step. |

More chapters are in progress on the site, starting with what happens once the loop is correct but the step is still slow: CUDA graphs, attention backends, speculative decoding, and then scaling past one GPU.

## How to read it

Open a chapter and press <kbd>→</kbd> to step, <kbd>←</kbd> to go back, and <kbd>space</kbd> to let it play. The last step of each chapter offers the next one. There is no scrolling: one picture per chapter, filling the viewport.

## How it is built

Astro with MDX for the pages, Svelte 5 for the visualizations, KaTeX for the math. Each visualization is a plain function of a step number: a small script of per-step states, rendered as SVG, with transitions between states. Written and drawn by [Wilson Zheng](https://wilsonzheng0327.github.io).

## License

This page is licensed under CC BY-NC-SA 4.0, the same as the course text.
