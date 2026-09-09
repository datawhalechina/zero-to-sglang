#!/usr/bin/env python3
"""Check course chapters against the writing templates.

    python3 scripts/check_chapters.py --all              # every chapter under course-material/
    python3 scripts/check_chapters.py FILE [FILE ...]    # only these files (CI passes the PR diff)

Rules are taken from course-material/ch/WRITING_TEMPLATE.md (Chinese chapters) and
course-material/eng/WRITING_TEMPLATE.md (English chapters):

  * file name        第N章_中文标题.md            /  ChapterN_English-Title.md
  * one H1           # 第 N 章 中文标题            /  # Chapter N Title
  * sections         ## N ...  ### N.M ...  #### N.M.K ...   numbered, continuous, nothing below H4
  * ending           ## K 总结与测试题 (K.1 课程总结, K.2 测试题) then ## 参考资料
                     ## K Summary and Exercises (K.1 Summary, K.2 Exercises) then ## References
  * references       a bullet list, one entry per line
  * images           <img src="./images/章号-序号-说明.png" width="800">
  * outline          chapter number within the part's chapter count
  * naming           no other inference projects named anywhere under course-material/

Only course-material/ is checked; community/ is out of scope for this script.

A chapter file whose only heading is the H1 is treated as a placeholder: only the
file name, the H1 and the naming rule are checked.

Only stdlib is used. Exit status is 1 when any error is found.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Chapter count per part, from the course outline in both templates.
PART_CHAPTERS = {1: 5, 2: 10, 3: 5, 4: 3}

# Other inference projects that must not be named in course text (template rule 1).
# Extend as needed; matching is case-insensitive and whole-word.
FORBIDDEN_PROJECTS = [
    "vllm",
    "tensorrt-llm",
    "trt-llm",
    "lmdeploy",
    "llama.cpp",
    "ollama",
    "text-generation-inference",
]
FORBIDDEN_RE = re.compile(
    r"(?<![A-Za-z0-9_])(" + "|".join(re.escape(p) for p in FORBIDDEN_PROJECTS) + r")(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

IMAGE_EXTS = r"(?:png|jpe?g|svg|gif|webp)"
IMAGE_NAME_RE = re.compile(r"^(\d+)-(\d+)-.+\." + IMAGE_EXTS + r"$", re.IGNORECASE)
IMG_TAG_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
ATTR_RE = re.compile(r"""(\w+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")


@dataclass(frozen=True)
class Lang:
    key: str
    root: str
    file_re: re.Pattern
    h1_re: re.Pattern
    summary: str
    summary_sub1: str
    summary_sub2: str
    references: str
    extra_image_dirs: tuple[str, ...]


LANGS = {
    "ch": Lang(
        key="ch",
        root="course-material/ch",
        file_re=re.compile(r"^第(\d+)章_.+\.md$"),
        h1_re=re.compile(r"^第 (\d+) 章 (.+)$"),
        summary="总结与测试题",
        summary_sub1="课程总结",
        summary_sub2="测试题",
        references="参考资料",
        extra_image_dirs=(),
    ),
    "eng": Lang(
        key="eng",
        root="course-material/eng",
        file_re=re.compile(r"^Chapter(\d+)_.+\.md$"),
        h1_re=re.compile(r"^Chapter (\d+) (.+)$"),
        summary="Summary and Exercises",
        summary_sub1="Summary",
        summary_sub2="Exercises",
        references="References",
        # The English template allows reusing a Chinese figure by relative path.
        extra_image_dirs=("../../ch/part{part}/images/",),
    ),
}

H2_RE = re.compile(r"^(\d+) (.+)$")
H3_RE = re.compile(r"^(\d+)\.(\d+) (.+)$")
H4_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+) (.+)$")


class Report:
    def __init__(self) -> None:
        self.errors = 0
        self.on_github = bool(os.environ.get("GITHUB_ACTIONS"))

    def error(self, path: Path, line: int, msg: str) -> None:
        self.errors += 1
        rel = path.relative_to(REPO) if path.is_absolute() else path
        print(f"{rel}:{line}: {msg}")
        if self.on_github:
            print(f"::error file={rel},line={line}::{msg}")


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def outside_code(lines: list[str]):
    """Yield (line_no, line) for lines that are not inside a fenced code block."""
    fence = None
    for i, line in enumerate(lines, 1):
        m = FENCE_RE.match(line)
        if m:
            marker = m.group(1)
            if fence is None:
                fence = marker
            elif marker == fence:
                fence = None
            continue
        if fence is None:
            yield i, line


def check_naming(path: Path, lines: list[str], rep: Report) -> None:
    for i, line in enumerate(lines, 1):
        m = FORBIDDEN_RE.search(line)
        if m:
            rep.error(path, i, f"names another inference project ({m.group(1)}); "
                               f"say 「其他推理引擎」/ 'other inference engines' instead")


def part_of(path: Path) -> int | None:
    m = re.fullmatch(r"part([1-4])", path.parent.name)
    return int(m.group(1)) if m else None


def check_images(path: Path, lang: Lang, part: int, chapter: int,
                 lines: list[str], rep: Report) -> None:
    allowed_prefixes = ["./images/", "images/"] + [d.format(part=part) for d in lang.extra_image_dirs]
    for i, line in outside_code(lines):
        if MD_IMAGE_RE.search(line):
            rep.error(path, i, 'use an HTML tag for images: <img src="./images/..." width="800">')
        for tag in IMG_TAG_RE.findall(line):
            attrs = {k.lower(): (a if a is not None else b if b is not None else c)
                     for k, a, b, c in ATTR_RE.findall(tag)}
            src = attrs.get("src")
            if not src:
                rep.error(path, i, "<img> without src")
                continue
            width = attrs.get("width")
            if width != "800":
                rep.error(path, i, f'image width must be 800 (got {width!r})')
            if not any(src.startswith(p) for p in allowed_prefixes):
                rep.error(path, i, f"image must live under this part's images/ directory (src={src})")
            name = src.rsplit("/", 1)[-1]
            m = IMAGE_NAME_RE.match(name)
            if not m:
                rep.error(path, i, f"image file name must be 章号-序号-说明.png, e.g. {chapter}-1-xxx.png (got {name})")
            elif int(m.group(1)) != chapter:
                rep.error(path, i, f"image file name should start with the chapter number {chapter}- (got {name})")
            if not (path.parent / src).exists():
                rep.error(path, i, f"image file not found: {src}")


def check_chapter(path: Path, lang: Lang, rep: Report) -> None:
    lines = read_lines(path)
    part = part_of(path)
    assert part is not None

    # --- file name
    m = lang.file_re.match(path.name)
    if not m:
        rep.error(path, 1, f"file name must match {lang.file_re.pattern}")
        return
    chapter = int(m.group(1))
    if not 1 <= chapter <= PART_CHAPTERS[part]:
        rep.error(path, 1, f"Part {part} has {PART_CHAPTERS[part]} chapters in the outline; chapter {chapter} is not one of them")

    # --- headings
    headings = []
    for i, line in outside_code(lines):
        hm = HEADING_RE.match(line)
        if hm:
            headings.append((i, len(hm.group(1)), hm.group(2).strip()))

    h1s = [h for h in headings if h[1] == 1]
    if len(h1s) != 1:
        rep.error(path, h1s[1][0] if len(h1s) > 1 else 1,
                  f"exactly one level-1 heading (the chapter title) is allowed, found {len(h1s)}")
    if h1s:
        i, _, text = h1s[0]
        hm = lang.h1_re.match(text)
        if not hm:
            rep.error(path, i, f"chapter title must look like: {example_h1(lang)}")
        elif int(hm.group(1)) != chapter:
            rep.error(path, i, f"chapter number in title ({hm.group(1)}) differs from file name ({chapter})")
        if headings and headings[0][1] != 1:
            rep.error(path, headings[0][0], "the chapter title must be the first heading")

    check_images(path, lang, part, chapter, lines, rep)

    h2s = [h for h in headings if h[1] == 2]
    if not h2s:
        # Placeholder chapter: nothing more to check.
        return

    # --- level-2 sections: numbered 1..K, then the unnumbered references section last
    last_i, _, last_text = h2s[-1]
    if last_text != lang.references:
        rep.error(path, last_i, f"the last level-2 heading must be '## {lang.references}'")
        numbered = h2s
    else:
        numbered = h2s[:-1]
    for idx, (i, _, text) in enumerate(h2s):
        if text == lang.references and idx != len(h2s) - 1:
            rep.error(path, i, f"'## {lang.references}' must be the last section")

    expected = 1
    for i, _, text in numbered:
        hm = H2_RE.match(text)
        if not hm:
            rep.error(path, i, f"level-2 heading must be numbered: '## {expected} ...'")
            continue
        n = int(hm.group(1))
        if n != expected:
            rep.error(path, i, f"level-2 numbering not continuous: expected {expected}, got {n}")
        expected = n + 1

    # --- summary section
    if numbered:
        si, _, stext = numbered[-1]
        sm = H2_RE.match(stext)
        if not sm or sm.group(2) != lang.summary:
            rep.error(path, si, f"the last numbered section must be '## K {lang.summary}'")
        else:
            k = sm.group(1)
            subs = [h for h in headings
                    if h[1] == 3 and si < h[0] < (last_i if last_text == lang.references else 10**9)]
            want = [f"{k}.1 {lang.summary_sub1}", f"{k}.2 {lang.summary_sub2}"]
            got = [h[2] for h in subs]
            if got != want:
                rep.error(path, si, f"'{lang.summary}' must contain exactly '### {want[0]}' and '### {want[1]}' (got {got})")

    # --- numbering of H3 / H4, depth limit, no headings under references
    cur_h2 = None      # int
    cur_h3 = None      # (h2, h3)
    next_h3 = 1
    next_h4 = 1
    in_refs = False
    for i, level, text in headings:
        if level == 1:
            continue
        if in_refs and level >= 2 and not (level == 2 and text == lang.references):
            rep.error(path, i, f"no headings allowed under '## {lang.references}'")
            continue
        if level == 2:
            if text == lang.references:
                in_refs = True
                cur_h2 = None
                continue
            hm = H2_RE.match(text)
            cur_h2 = int(hm.group(1)) if hm else None
            cur_h3 = None
            next_h3 = 1
        elif level == 3:
            if cur_h2 is None:
                rep.error(path, i, "level-3 heading without a numbered level-2 section above it")
                continue
            hm = H3_RE.match(text)
            if not hm:
                rep.error(path, i, f"level-3 heading must be numbered: '### {cur_h2}.{next_h3} ...'")
                cur_h3 = None
                continue
            a, b = int(hm.group(1)), int(hm.group(2))
            if a != cur_h2:
                rep.error(path, i, f"level-3 heading {a}.{b} is under section {cur_h2}")
            elif b != next_h3:
                rep.error(path, i, f"level-3 numbering not continuous: expected {cur_h2}.{next_h3}, got {a}.{b}")
            cur_h3 = (a, b)
            next_h3 = b + 1
            next_h4 = 1
        elif level == 4:
            if cur_h3 is None:
                rep.error(path, i, "level-4 heading without a numbered level-3 heading above it")
                continue
            hm = H4_RE.match(text)
            if not hm:
                rep.error(path, i, f"level-4 heading must be numbered: '#### {cur_h3[0]}.{cur_h3[1]}.{next_h4} ...'")
                continue
            a, b, c = (int(g) for g in hm.groups()[:3])
            if (a, b) != cur_h3:
                rep.error(path, i, f"level-4 heading {a}.{b}.{c} is under {cur_h3[0]}.{cur_h3[1]}")
            elif c != next_h4:
                rep.error(path, i, f"level-4 numbering not continuous: expected {a}.{b}.{next_h4}, got {a}.{b}.{c}")
            next_h4 = c + 1
        else:
            rep.error(path, i, "headings deeper than level 4 are not allowed; use a list or a bold line")

    # --- references body: bullet list only
    if last_text == lang.references:
        for i, line in outside_code(lines):
            if i <= last_i or not line.strip():
                continue
            if not re.match(r"^\s*[-*] ", line):
                rep.error(path, i, f"'{lang.references}' must be a bullet list, one entry per line")


def example_h1(lang: Lang) -> str:
    return "# 第 2 章 推理入门" if lang.key == "ch" else "# Chapter 2 Introduction to Inference"


def lang_of(path: Path) -> Lang | None:
    try:
        rel = path.resolve().relative_to(REPO)
    except ValueError:
        return None
    if len(rel.parts) < 2 or rel.parts[0] != "course-material":
        return None
    return LANGS.get(rel.parts[1])


def is_chapter_file(path: Path) -> bool:
    return path.suffix == ".md" and part_of(path) is not None and path.parent.parent.name in LANGS


def collect_all() -> list[Path]:
    files = []
    for lang in LANGS.values():
        files.extend(sorted((REPO / lang.root).rglob("*.md")))
    return files


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="markdown files to check (default: none)")
    ap.add_argument("--all", action="store_true", help="check every markdown file under course-material/")
    args = ap.parse_args(argv)

    paths = collect_all() if args.all else [Path(f).resolve() for f in args.files]
    rep = Report()
    checked = 0
    for path in paths:
        if not path.exists():
            if not args.all:
                rep.error(path, 1, "file not found (was it passed with git's quoted path? use -c core.quotePath=false)")
            continue
        if path.suffix != ".md":
            continue
        lang = lang_of(path)
        if lang is None or path.name == "WRITING_TEMPLATE.md":
            continue
        checked += 1
        check_naming(path, read_lines(path), rep)
        if is_chapter_file(path):
            check_chapter(path, lang, rep)

    print(f"\nchecked {checked} file(s), {rep.errors} error(s)")
    return 1 if rep.errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
