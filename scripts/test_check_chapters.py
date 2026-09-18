"""Regression tests for chapter opening and prose indentation rules."""

import contextlib
import io
import unittest
from unittest.mock import patch

from check_chapters import (
    LANGS,
    REPO,
    Report,
    check_chapter,
    check_opening,
    check_paragraphs,
    is_placeholder,
    outside_code,
)


class ChapterLayoutTests(unittest.TestCase):
    def check_text(self, check, text: str, lang: str = "eng", part: int = 1) -> int:
        path = REPO / "course-material" / lang / f"part{part}" / "Chapter1_Example.md"
        report = Report()
        with contextlib.redirect_stdout(io.StringIO()):
            check(path, text.splitlines(), report)
        return report.errors

    def test_introduction_and_objectives_in_both_languages(self):
        for lang, title in (("ch", "本章学习目标"), ("eng", "Learning Objectives")):
            with self.subTest(lang=lang):
                text = f"# Chapter 1 Example\n\nA prose introduction.\n\n## 1 {title}\n"
                self.assertEqual(self.check_text(check_opening, text, lang), 0)

    def test_introduction_cannot_be_replaced_by_other_blocks(self):
        for block in ("", "- Objective", "> Quote", "![Image](img.png)",
                      "<!-- Comment -->", "```python\nprint(1)\n```", "$$x$$", "---"):
            with self.subTest(block=block):
                text = f"# Chapter 1 Example\n\n{block}\n\n## 1 Learning Objectives\n"
                self.assertGreater(self.check_text(check_opening, text), 0)

    def test_missing_or_wrong_objectives(self):
        for section in ("", "## 1 Overview", "## 2 Learning Objectives"):
            with self.subTest(section=section):
                text = f"# Chapter 1 Example\n\nIntroduction.\n\n{section}\n"
                self.assertGreater(self.check_text(check_opening, text), 0)

    def test_part0_needs_introduction_but_not_objectives(self):
        text = "# Part 0.1 Example\n\nIntroduction.\n\n## Coding ethics\n"
        self.assertEqual(self.check_text(check_opening, text, part=0), 0)
        self.assertGreater(self.check_text(check_opening, text.replace("Introduction.\n\n", ""), part=0), 0)

    def test_only_standard_notices_are_placeholders(self):
        for notice in ("本章内容撰写中，敬请期待。", "This chapter is being written. Stay tuned."):
            text = f"# Chapter 1 Example\n\n{notice}\n"
            self.assertTrue(is_placeholder(text.splitlines()))
            self.assertEqual(self.check_text(check_opening, text), 0)
            self.assertFalse(is_placeholder((text + "\nActual content.\n").splitlines()))
        self.assertFalse(is_placeholder(["# Chapter 1 Example", "", "Actual content."]))

    def test_plain_indentation_and_encoded_spaces_are_rejected(self):
        for prefix in (" ", "  ", "    ", "\t", "\u3000", "\u2003", "\u00a0",
                       "&emsp;", "&ensp;", "&nbsp;", "&#8195;", "&#x2003;", "&#32;"):
            with self.subTest(prefix=prefix):
                self.assertGreater(self.check_text(check_paragraphs, prefix + "Prose."), 0)

    def test_entities_in_lists_quotes_and_html_are_rejected(self):
        for line in ("- &emsp;Prose", "> &nbsp;Prose", '<p>&#x2003;Prose</p>',
                     '<p style="text-indent: 2em">Prose</p>'):
            with self.subTest(line=line):
                self.assertGreater(self.check_text(check_paragraphs, line), 0)

    def test_structural_indentation_is_preserved(self):
        text = """# Chapter 1 Example

Introduction.

- Item
  - Nested item
    Continuation.

  Another paragraph in the parent item.

1. Ordered item
   > Hint for this item.

> Quote.
> > Nested quote.

<div align="center">
  <img src="image.png" width="800">
  <p>Caption.</p>
</div>

$$
  x = y
$$

```python
    indented_code()
&emsp;literal_example
```

> ```text
> &emsp;literal_example
> ```
"""
        self.assertEqual(self.check_text(check_paragraphs, text), 0)

    def test_top_level_prose_after_list_is_checked(self):
        text = "- Item\n  Continuation.\n\nNormal paragraph.\n\n  Indented paragraph.\n"
        self.assertEqual(self.check_text(check_paragraphs, text), 1)

    def test_long_fence_is_not_closed_by_shorter_fence(self):
        lines = ["````markdown", "```python", "  code", "```", "````", "Prose"]
        self.assertEqual(list(outside_code(lines)), [(6, "Prose")])

    def test_heading_spacing(self):
        self.assertGreater(self.check_text(check_paragraphs, "Text\n## 1 Heading\nText"), 0)
        self.assertEqual(self.check_text(check_paragraphs, "Text\n\n## 1 Heading\n\nText"), 0)

    def test_headingless_body_reports_error_without_crashing(self):
        path = REPO / "course-material/eng/part1/Chapter1_Example.md"
        report = Report()
        with patch("check_chapters.read_lines", return_value=["# Chapter 1 Example", "", "Body."]):
            with contextlib.redirect_stdout(io.StringIO()):
                check_chapter(path, LANGS["eng"], report)
        self.assertGreater(report.errors, 0)


if __name__ == "__main__":
    unittest.main()
