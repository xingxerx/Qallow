#!/usr/bin/env python3
"""
Qallow Bend conversion scaffold.

Reads C / CUDA translation units and emits Bend template files under bend/auto/.
Each generated Bend file contains a header referencing the source file and
stubbed function definitions that mirror the original signatures.  This gives
the follow-up conversion pipeline deterministic anchors to fill in with the
real Bend logic.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, List, Tuple

ROOT = Path(__file__).resolve().parent
SRC_DIRS = [
    ROOT / "backend" / "cpu",
    ROOT / "backend" / "cuda",
    ROOT / "interface",
]
OUT_ROOT = ROOT / "bend" / "auto"

# Cache simple type substitutions for doc output
TYPE_MAP = {
    "int": "Int",
    "float": "F32",
    "double": "F64",
    "bool": "Bool",
    "size_t": "Int",
    "uint32_t": "U32",
    "uint64_t": "U64",
    "char": "U8",
    "const": "",
    "void": "Unit",
}

FUNC_PATTERN = re.compile(
    r"""
    ^                                   # start of line
    (?P<header>
        (?:static\s+)?                  # optional static
        (?:inline\s+)?                  # optional inline
        [\w\*\s]+?                      # return type
    )
    (?P<name>[A-Za-z_]\w*)              # function name
    \s*
    \(
        (?P<args>[^)]*)
    \)
    \s*
    (?P<trailer>\{|;)                   # function body or prototype
    """,
    re.MULTILINE | re.VERBOSE,
)


def discover_sources() -> Iterable[Path]:
    for src_root in SRC_DIRS:
        if not src_root.exists():
            continue
        for path in src_root.glob("**/*"):
            if path.suffix not in (".c", ".cu"):
                continue
            if path.name.endswith((".obj", ".o", ".dup", ".backup")):
                continue
            yield path.relative_to(ROOT)


def parse_functions(text: str) -> List[Tuple[str, List[str]]]:
    functions: List[Tuple[str, List[str]]] = []
    for match in FUNC_PATTERN.finditer(text):
        if match.group("trailer") == ";":
            # skip prototypes / externs
            continue
        name = match.group("name")
        args_raw = match.group("args").strip()
        args: List[str] = []
        if args_raw and args_raw != "void":
            for arg in args_raw.split(","):
                arg = arg.strip()
                if not arg:
                    continue
                parts = arg.split()
                arg_name = parts[-1].replace("*", "").replace("&", "")
                if arg_name.startswith("*"):
                    arg_name = arg_name.strip("*")
                args.append(arg_name or f"arg_{len(args)}")
        functions.append((name, args))
    return functions


def bend_signature(name: str, args: List[str]) -> str:
    arg_list = ", ".join(args) if args else ""
    return f"def {name}({arg_list}):"


def write_stub(src_rel: Path) -> None:
    src_path = ROOT / src_rel
    out_path = OUT_ROOT / src_rel.with_suffix(".bend")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    functions = parse_functions(src_path.read_text(encoding="utf-8", errors="ignore"))
    lines: List[str] = []
    lines.append("# Auto-generated Bend template")
    lines.append(f"# Source: {src_rel.as_posix()}")
    lines.append("#")
    lines.append("# TODO: Replace the stub bodies with translated Bend logic.")
    lines.append("")

    if not functions:
        lines.append("# NOTE: No functions detected in source file.")
    else:
        for name, args in functions:
            lines.append(bend_signature(name, args))
            lines.append("  # TODO: implement")
            lines.append("  pass")
            lines.append("")

    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    sources = sorted(discover_sources())
    if not sources:
        print("No source files discovered; nothing to convert.")
        return

    for src in sources:
        write_stub(src)
    print(f"Generated Bend stubs for {len(sources)} translation units under {OUT_ROOT}")


if __name__ == "__main__":
    main()
