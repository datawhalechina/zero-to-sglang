#!/usr/bin/env python3
"""Check that the site's navigation, pages and links are consistent.

    python3 scripts/check_site.py                       # source checks
    python3 scripts/check_site.py --dist docs/.vitepress/dist   # + checks on a finished build

Source checks:
  * every nav / sidebar / locale link in docs/.vitepress/config.mts points to an existing page
  * every chapter page under ch/part*/ and eng/part*/ is reachable from the sidebar (no orphans)
  * every relative link, href and src in README.md, README_en.md, ch/**/*.md, eng/**/*.md
    resolves to an existing file (links inside fenced code blocks are ignored)

Build checks (--dist):
  * the site root, /ch/ and /eng/ home pages were emitted
  * every markdown page under ch/ and eng/ produced an HTML file
  * every Chinese page has a redirect stub at its pre-move root-level URL

Only stdlib is used. Exit status is 1 when any error is found.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "docs" / ".vitepress" / "config.mts"
LOCALES = ("ch", "eng")

CONFIG_LINK_RE = re.compile(r"link:\s*'(/[^']*)'")
MD_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
ATTR_LINK_RE = re.compile(r"""\b(?:href|src)\s*=\s*["']([^"']+)["']""")
YAML_LINK_RE = re.compile(r"^\s*link:\s*(/\S+)\s*$")
EXTERNAL_RE = re.compile(r"^(https?:|mailto:|tel:|#|data:|javascript:)", re.IGNORECASE)
FENCE_RE = re.compile(r"^\s*(```|~~~)")


class Report:
    def __init__(self) -> None:
        self.errors = 0
        self.on_github = bool(os.environ.get("GITHUB_ACTIONS"))

    def error(self, path: Path | str, line: int, msg: str) -> None:
        self.errors += 1
        rel = path.relative_to(REPO) if isinstance(path, Path) and path.is_absolute() else path
        print(f"{rel}:{line}: {msg}")
        if self.on_github:
            print(f"::error file={rel},line={line}::{msg}")


def outside_code(lines: list[str]):
    fence = None
    for i, line in enumerate(lines, 1):
        m = FENCE_RE.match(line)
        if m:
            if fence is None:
                fence = m.group(1)
            elif m.group(1) == fence:
                fence = None
            continue
        if fence is None:
            yield i, line


def site_link_to_file(link: str) -> Path:
    """'/ch/part0/X' -> ch/part0/X.md, '/ch/' -> ch/index.md, '/ch/X.html' -> ch/X.md"""
    path = link.split("#")[0].split("?")[0].lstrip("/")
    if path.endswith("/") or path == "":
        return REPO / path / "index.md"
    if path.endswith(".html"):
        path = path[: -len(".html")] + ".md"
    elif not Path(path).suffix:
        path += ".md"
    return REPO / path


def content_pages() -> list[Path]:
    pages = []
    for loc in LOCALES:
        pages.extend(p for p in sorted((REPO / loc).rglob("*.md")) if p.name != "WRITING_TEMPLATE.md")
    return pages


def chapter_pages() -> list[Path]:
    return [p for p in content_pages() if re.fullmatch(r"part\d", p.parent.name)]


def check_config(rep: Report) -> set[Path]:
    """Every link in config.mts resolves. Returns the set of linked page files."""
    linked: set[Path] = set()
    for i, line in enumerate(CONFIG.read_text(encoding="utf-8").splitlines(), 1):
        for link in CONFIG_LINK_RE.findall(line):
            target = site_link_to_file(link)
            linked.add(target)
            if not target.exists():
                rep.error(CONFIG, i, f"nav/sidebar link {link} has no page ({target.relative_to(REPO)})")
    return linked


def check_orphans(linked: set[Path], rep: Report) -> None:
    for page in chapter_pages():
        if page not in linked:
            rep.error(page, 1, "chapter page is not listed in the sidebar (docs/.vitepress/config.mts)")


def resolve_relative(src: Path, target: str) -> Path:
    t = target.split("#")[0].split("?")[0]
    if t.startswith("/"):
        return site_link_to_file(t)
    return (src.parent / t).resolve()


def check_links(rep: Report) -> None:
    files = [REPO / "README.md", REPO / "README_en.md"]
    for loc in LOCALES:
        files.extend(sorted((REPO / loc).rglob("*.md")))
    for f in files:
        if not f.exists():
            continue
        lines = f.read_text(encoding="utf-8").splitlines()
        for i, line in outside_code(lines):
            targets = MD_LINK_RE.findall(line) + ATTR_LINK_RE.findall(line)
            ym = YAML_LINK_RE.match(line)
            if ym:
                targets.append(ym.group(1))
            for t in targets:
                if EXTERNAL_RE.match(t) or t.startswith("<"):
                    continue
                bare = t.split("#")[0].split("?")[0]
                if not bare:
                    continue
                resolved = resolve_relative(f, t)
                if resolved.exists() or resolved.with_suffix(".md").exists():
                    continue
                rep.error(f, i, f"broken link: {t}")


def check_dist(dist: Path, rep: Report) -> None:
    if not dist.is_dir():
        rep.error(dist, 1, "build output directory not found")
        return
    for must in ("index.html", "ch/index.html", "eng/index.html"):
        if not (dist / must).exists():
            rep.error(dist / must, 1, "expected build output is missing")
    for page in content_pages():
        rel = page.relative_to(REPO).with_suffix(".html")
        if not (dist / rel).exists():
            rep.error(page, 1, f"page was not built: {rel}")
        if rel.parts[0] == "ch":
            stub = dist / Path(*rel.parts[1:])
            if not stub.exists():
                rep.error(page, 1, f"missing redirect stub for the old URL: {Path(*rel.parts[1:])}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dist", type=Path, help="also check a finished VitePress build in this directory")
    args = ap.parse_args(argv)

    rep = Report()
    linked = check_config(rep)
    check_orphans(linked, rep)
    check_links(rep)
    if args.dist:
        check_dist(args.dist.resolve(), rep)

    print(f"\nsite check: {rep.errors} error(s)")
    return 1 if rep.errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
