#!/usr/bin/env python3
"""Validate and optionally restore the canonical internal CI workflow.

This helper keeps `.github/workflows/internal-ci.yml` in sync with the
reference pipeline we maintain in-repo.  It is meant to be run in CI as
well as locally before pushing workflow changes.
"""


import argparse
import base64
import difflib
import sys
import zlib
from pathlib import Path
from typing import List, Sequence, Tuple

WORKFLOW_PATH = Path(".github/workflows/internal-ci.yml")

CANONICAL_PIPELINE_B64 = (
    "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"
)


REQUIRED_SNIPPETS: Sequence[Tuple[str, str]] = (
    ("internal CI validator", "python3 scripts/check_internal_ci_pipeline.py"),
    ("human approval validation", "python3 scripts/verify_human_approval.py"),
    ("CPU configure flags", "cmake -S . -B build/CPU -DQALLOW_ENABLE_CUDA=OFF"),
    ("Accelerator smoke execution", "./build/CPU/qallow_unified_cpu run --accelerator"),
    ("Rust release build", "cargo build --release"),
)


def _canonical_workflow() -> str:
    """Return the canonical workflow text with a trailing newline."""
    content = zlib.decompress(base64.b64decode(CANONICAL_PIPELINE_B64)).decode("utf-8")
    if not content.endswith("\n"):
        content += "\n"
    return content


def _validate_content(current: str) -> List[str]:
    """Return validation failures for the given workflow content."""
    failures: List[str] = []
    expected = _canonical_workflow()

    if current != expected:
        failures.append("Workflow content diverges from the canonical template.")

    for label, snippet in REQUIRED_SNIPPETS:
        if snippet not in current:
            failures.append(f"Missing required snippet '{label}': {snippet}")

    return failures


def _show_diff(current: str, expected: str) -> None:
    diff = difflib.unified_diff(
        current.splitlines(keepends=True),
        expected.splitlines(keepends=True),
        fromfile=str(WORKFLOW_PATH),
        tofile="canonical-internal-ci.yml",
    )
    sys.stdout.writelines(diff)


def _write_canonical() -> None:
    expected = _canonical_workflow()
    WORKFLOW_PATH.write_text(expected, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ensure the internal CI workflow matches the canonical template."
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Rewrite the workflow file with the canonical template.",
    )
    parser.add_argument(
        "--show-diff",
        action="store_true",
        help="Display a unified diff when validation fails.",
    )
    args = parser.parse_args(argv)

    if not WORKFLOW_PATH.exists():
        if args.fix:
            print(f"[check-internal-ci] {WORKFLOW_PATH} missing; creating from template.")
            WORKFLOW_PATH.parent.mkdir(parents=True, exist_ok=True)
            _write_canonical()
            return 0
        print(f"[check-internal-ci] ERROR: {WORKFLOW_PATH} is missing.", file=sys.stderr)
        return 1

    current = WORKFLOW_PATH.read_text(encoding="utf-8")
    failures = _validate_content(current)

    if not failures:
        print("[check-internal-ci] Workflow matches canonical template.")
        return 0

    if args.fix:
        print("[check-internal-ci] Applying canonical workflow template.")
        _write_canonical()
        return 0

    print("[check-internal-ci] ERROR: workflow validation failed:", file=sys.stderr)
    for failure in failures:
        print(f"  - {failure}", file=sys.stderr)

    if args.show_diff:
        _show_diff(current, _canonical_workflow())

    return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
