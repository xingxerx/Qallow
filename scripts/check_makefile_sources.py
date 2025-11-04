# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] #!/usr/bin/env python3
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] """Ensure critical C sources are included in Makefile builds."""
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] from pathlib import Path
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] REPO_ROOT = Path(__file__).resolve().parent.parent
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # Source files that are intentionally excluded from the unified build.
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] SKIP_SOURCES: set[Path] = set()
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] def expand_make_variable(var_name: str) -> list[str]:
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]     make_stub = f"include Makefile\nprint::\n\t@printf '%s\\n' \"$({var_name})\"\n"
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]     try:
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]         result = subprocess.run(
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]             ["make", "-s", "-f", "-", "ACCELERATOR=CPU", "print"],
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]             input=make_stub,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        sys.stderr.write("Failed to evaluate Makefile variable.\n")
        sys.stderr.write(exc.stderr)
        raise
    return result.stdout.strip().split()


def main() -> int:
    obj_files = {Path(entry) for entry in expand_make_variable("OBJ_C")}
    if not obj_files:
        sys.stderr.write("OBJ_C variable expansion returned no entries.\n")
        return 1

    missing = []
    for source_path in sorted(REPO_ROOT.joinpath("src").rglob("*.c")):
        rel_path = source_path.relative_to(REPO_ROOT)
        if rel_path in SKIP_SOURCES:
            continue
        expected_obj = Path("build/CPU") / rel_path.with_suffix(".o")
        if expected_obj not in obj_files:
            missing.append(rel_path.as_posix())

    if missing:
        sys.stderr.write("The following src/ C files are not covered by OBJ_C in Makefile:\n")
        for rel in missing:
            sys.stderr.write(f"  - {rel}\n")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
