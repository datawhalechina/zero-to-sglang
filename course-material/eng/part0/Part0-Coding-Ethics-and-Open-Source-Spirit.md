# Part 0 — Before you learn

## 0.1 Coding ethics and open-source spirit

Welcome to zero-to-sglang. This first lesson is not about technology. There is an old saying that character comes before craft, and before we teach you anything technical we want to pass on a healthy open-source spirit, and show how to communicate and collaborate with others in an open-source community. In the AI era, writing code has become far easier than reviewing it, so we hope every open-source contributor takes responsibility for their own code, respects the craft, and respects every reviewer's time.

What follows may look like common sense. But every point here comes from real cases we ran into while maintaining open-source communities, so we are spelling them out in the hope that they become second nature.

---

## Coding ethics

### 1. Take responsibility for your own code

First, to be clear: the open-source community is not against writing code with AI. Every programmer does it. But for anything headed to production, we expect that the author has carefully read the code they submit and knows why it is written the way it is.

Code adapted from Stack Overflow the old-fashioned way, code written by AI, code typed by hand: all of it is legitimate. What has changed in the AI era is the cost. 500 lines used to take an hour, sometimes an afternoon; now it takes 30 seconds. But a reviewer still needs a lot of time to read those 500 lines and decide whether they are correct. This is an invisible transfer of workload and responsibility, and it is exactly why every contributor needs the self-discipline to own their code. Otherwise the project cannot keep going.

So before opening a PR, ask yourself a few questions:

- What problem does this PR solve? Is there a clear motivation?
- What does this PR deliver, and how can someone reproduce the result?
- Which code regions and components does it touch? What is the call flow? How invasive is it?
- Pick any function in the diff at random: can I say what would happen if it were deleted?

If you cannot answer any one of these, do not submit yet. Keep PRs from growing too large and make sure you fully understand them. That lowers the reviewer's cost, and the cost of the back-and-forth that follows, which brings us to the second point.

### 2. Communicate like a human

Issues, PR descriptions, review replies, chat in the group: write them in your own words. Do not paste AI replies wholesale.

The reason is the same as above: respect other people's time. Everyone can spot an AI-generated reply by now. In two hundred words there may be fifteen words of actual information; the rest is AI packaging.

A typical example:

> Thank you for your review! The point you raised is very valuable. Regarding whether to enable CUDA Graph, I have conducted an in-depth analysis. 1. Performance: CUDA Graph can effectively reduce the CPU overhead of kernel launches, with significant gains in small-batch scenarios... 2. Memory: the additional memory consumed by graph capture needs to be considered... I ran experiments, and the results are roughly... In summary, if the goal is peak performance, I recommend enabling CUDA Graph; if the goal is minimal memory, I recommend disabling it. I believe the current implementation is a reasonable trade-off. I look forward to your further guidance...

> Tried it. With CUDA Graph on, the decode step at batch=n drops from xx ms to xx ms; profiles from before and after are attached below. Graph capture costs an extra yy of memory.

The second is short and clear; you know what it says at a glance. The first is classic AI slop, generated on the spot, and it inspires very little confidence.

Worse, when a maintainer sees an AI-flavored reply, the first thought is "this PR was probably generated too, and the author never read it." That directly damages the trust others place in your code.

### 3. Profile first, always

Profiling is always the first step of any performance work.

The right order is: profile, find the bottleneck, change it, profile again, compare. We want to avoid shooting the arrow first and painting the target around it. Landing an optimization and merging it into the repo is not the end of the story. In open-source practice many people make the same mistake: the part they set out to optimize accounts for maybe 3% of total time. Optimize a 3% piece down to zero and the end-to-end result is only 3% faster.

So before you start, answer one question: how much time does the code I am about to change take under a real workload? If you do not know, go measure it. That one hour can save you two weeks.

This is also why we say: do not chase merged PRs just to pad your project experience. The more people approach it that way, the more junk PRs like the ones described above pile up, and the good PRs drown among them.

We therefore ask that a performance PR include at least these three things:

1. **Before-and-after numbers.**
2. **A reproducible command.**
3. **Profiling evidence.** What people want to see is not "12% faster" but "why it is faster." Use a profiler, plot the timeline, learn to read and explain it, and make sure the PR actually solves the problem stated in the motivation.

### 4. Summary: what kind of first PR is unwelcome

Pulling the discussion together, here are the common pitfalls of a bad PR. Please check yourself against them before submitting:

1. A large AI-generated PR the author has not read. Thousands of lines, and questions about any part of it go unanswered.
2. A performance PR with no data and no reproducible results.
3. A refactor of core modules without design work, far too invasive.
4. Design decisions made without discussion: adding a new dependency, introducing a new abstraction, changing default parameters.
5. Ten spam PRs opened at once, none with a clear motivation.

So what does a good first PR look like? Small, complete, solving a real existing issue, with verification. Our usual advice: understand first, then ask, then contribute. A good PR author is, first of all, someone who can find a problem in the system and file an issue about it.

---

## Open-source spirit

### One for all, all for one

Everything this course relies on, PyTorch, FlashAttention, SGLang, the open weights you download, and this course itself, is open source. In other words, someone gave it to you for free. So the default mindset in open source is simple: you benefit from what others left behind, so leave your own work behind too, and keep the cycle turning. Concretely, for this course: you will probably hit plenty of pitfalls while building mini-sglang later, and your deployment environment will have problems. Spend ten minutes writing the pitfall up clearly and posting it, and the next person saves hours. That is all the open-source spirit is. And contribution is not only code. Documentation, translation, reproducing benchmarks, helping pin down an issue: these matter as much as writing kernels.

The open-source community welcomes people who want to learn the technology and are willing to contribute to it, because everyone starts out not knowing. Who it does not welcome are those who treat open source as a résumé-padding tool. Participating in open source for your résumé is not a problem in itself; nearly everyone's motivation includes some of that. The problem is confusing means and ends, when the goal becomes PR count rather than solving problems.

This lesson closes on three points: respect other people's time, respect the craft, and take responsibility for your own code. Writing code is not like building a bridge, where a collapse brings legal liability. But as fellow engineers, we hope everyone in this course carries an engineer's pride and sense of responsibility, and truly lives out "one for all, all for one" in the open-source community.

Character before craft. Please keep this open-source spirit with you as you move into the code that follows.
