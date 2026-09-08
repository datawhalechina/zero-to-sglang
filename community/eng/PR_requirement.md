# Community PR Requirements

Go through every item before you submit. PRs that do not meet them will be sent back.

1. Put the file in the matching directory under `community/eng/`: `builds/` for mini-sglang build logs, `pitfalls/` for traps and warnings, `notes/` for study notes. Name the file in English so the topic is clear at a glance.
2. State your environment at the top: GPU model, driver and CUDA version, SGLang version (tag or commit), Python and PyTorch versions. Pitfall reports cannot be reproduced without it.
3. Put images in the directory's `images/` folder and reference them with an HTML tag, width 800.
4. Logs, error messages and commands go in code blocks, never screenshots.
5. Do not name other inference projects in the text, figures or references, and do not compare against their numbers. When a comparison is needed, say "other inference engines".
6. The content must be something you verified yourself. No pure reposts, no pure translations, and nothing that duplicates the course text (fix the course text directly instead).
7. Write in your own words. AI can help you organize, but read the whole thing before you post it and make sure you stand behind every sentence. Same for the PR description: what changed and why, not a pasted block of AI output.
8. List only references you actually read and used.
9. In the same PR, update the "Accepted" table in [index.md](./index.md): add a row with date, directory, title and author.
10. Submitting means you agree to license the content under CC BY-NC-SA 4.0, the same as the course text.
