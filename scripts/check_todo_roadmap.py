#!/usr/bin/env python3
"""
Verify that newly introduced TODO-style comments are tracked in docs/TODO_ROADMAP.md.

By default the script looks at the staged diff (like a pre-commit hook).  In CI you
can point it at a base reference with `--base-ref origin/main` so that only additions
since the merge base are considered.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROADMAP = REPO_ROOT / "docs" / "TODO_ROADMAP.md"
MARKER_PATTERN = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b")
INCLUDE_SUFFIXES = {".c", ".h", ".py"}
EXCLUDE_DIRS = {"venv", ".git", "build", "__pycache__", ".mypy_cache"}


@dataclass
class Marker:
    path: Path
    text: str

    def format(self) -> str:
        return f"{self.path}:{self.text.strip()}"


def _run_git_command(args: Sequence[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=REPO_ROOT).decode("utf-8", "replace")
    except subprocess.CalledProcessError as exc:
        print(f"[todo-roadmap] git command failed: {' '.join(args)}", file=sys.stderr)
        raise SystemExit(exc.returncode)


def _should_consider(path: Path) -> bool:
    if path.suffix not in INCLUDE_SUFFIXES:
        return False
    return not any(part in EXCLUDE_DIRS for part in path.parts)


def _scan_diff(base_ref: str | None) -> List[Marker]:
    if base_ref:
        diff_args = ["git", "diff", "--unified=0", base_ref]
    else:
        diff_args = ["git", "diff", "--cached", "--unified=0"]
    diff_args.append("--")
    diff_output = _run_git_command(diff_args)
    markers: List[Marker] = []
    current_file: Path | None = None

    for line in diff_output.splitlines():
        if line.startswith("diff --git"):
            current_file = None
        elif line.startswith("+++ "):
            if line.startswith("+++ b/"):
                rel_path = line[6:]
                if rel_path == "/dev/null":
                    current_file = None
                else:
                    current_file = Path(rel_path)
            else:
                current_file = None
        elif not current_file:
            continue
        elif line.startswith("@@"):
            continue
        elif line.startswith("+++"):
            continue
        elif line.startswith("+"):
            if current_file.suffix not in INCLUDE_SUFFIXES:
                continue
            if not _should_consider(current_file):
                continue
            text = line[1:]
            if MARKER_PATTERN.search(text):
                markers.append(Marker(current_file, text))
    return markers


def _scan_repository(full: bool) -> List[Marker]:
    if not full:
        return []
    markers: List[Marker] = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(REPO_ROOT)
        if not _should_consider(rel):
            continue
        try:
            contents = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for line in contents.splitlines():
            if MARKER_PATTERN.search(line):
                markers.append(Marker(rel, line))
    return markers


def _load_roadmap(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Roadmap file missing: {path}")
    return path.read_text(encoding="utf-8", errors="ignore")


def _markers_not_in_roadmap(markers: Iterable[Marker], roadmap_text: str) -> List[Marker]:
    uncovered: List[Marker] = []
    for marker in markers:
        marker_text = marker.text.strip()
        rel_path = str(marker.path).replace("\\", "/")
        if marker_text and marker_text in roadmap_text:
            continue
        if rel_path in roadmap_text:
            continue
        uncovered.append(marker)
    return uncovered


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ensure TODO/FIXME/XXX/HACK additions are documented in TODO_ROADMAP.md"
    )
    parser.add_argument(
        "--roadmap",
        type=Path,
        default=DEFAULT_ROADMAP,
        help="Path to roadmap file (default: %(default)s)",
    )
    parser.add_argument(
        "--base-ref",
        help="Git reference to diff against (e.g. origin/main). "
        "If omitted, the script inspects the staged diff.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Additionally scan the entire repository (useful for CI baselines).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    try:
        roadmap_text = _load_roadmap(args.roadmap)
    except FileNotFoundError as exc:
        print(f"[todo-roadmap] ❌ {exc}", file=sys.stderr)
        return 1

    markers = _scan_diff(args.base_ref)
    if args.full:
        markers.extend(_scan_repository(full=True))

    if not markers:
        print("[todo-roadmap] ✅ No new TODO-style markers detected.")
        return 0

    uncovered = _markers_not_in_roadmap(markers, roadmap_text)
    if not uncovered:
        print("[todo-roadmap] ✅ All TODO markers are covered by the roadmap.")
        return 0

    print("[todo-roadmap] ❌ The following TODO-style markers are missing from the roadmap:")
    for marker in uncovered:
        print(f"  - {marker.format()}")

    print(
        f"[todo-roadmap] Please document these items in {args.roadmap} "
        "or remove the markers before committing."
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
