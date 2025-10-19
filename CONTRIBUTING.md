# Contributing to Qallow

Thanks for your interest in advancing Qallow's ethics-first autonomous runtime. This guide explains how to set up your environment, follow the coding standards, and propose changes.

## Prerequisites

- CMake ≥ 3.20 and Ninja or Make
- GCC ≥ 11 (or Clang ≥ 15)
- CUDA Toolkit ≥ 12.0 (optional)
- Python ≥ 3.10
- `clang-format` 15, `cmake-format`, `markdownlint`, `pre-commit`

Run `pip install -r python/requirements-dev.txt` (create if missing) for linting helpers. Install GPU drivers if you plan to run CUDA demos.

## Repository Workflow

- Fork the repo or create a feature branch from `main` (`git checkout -b feature/<topic>`).
- Keep commits focused and use Conventional Commit style (`feat:`, `fix:`, `docs:`, etc.).
- Ensure your branch merges cleanly with `main` before opening a pull request.
- Reference GitHub issues in commit messages and PR descriptions when applicable.

## Coding Standards

- C modules target C11, CUDA modules use CUDA C++ 17, support both GCC and NVCC builds.
- Place headers under `include/qallow/` (exported API) or `core/include/` (internal).
- Use the logging API (`qallow_log_*`) instead of `printf`/`fprintf`.
- Wrap expensive sections with `QALLOW_PROFILE_SCOPE("label")` for profiling.
- Keep functions under 120 lines when possible; factor reusable pieces into helpers.
- Ensure telemetry data conforms to the schema in `docs/QUICKSTART.md`.

## Testing

- Configure via `./scripts/build_all.sh` or `cmake -S . -B build`.
- Run unit and integration tests: `ctest --test-dir build`.
- Provide new tests for bug fixes or features in `tests/unit` or `tests/integration`.
- GPU changes require running `ctest -R cuda` and providing Nsight screenshots if possible.

## Documentation

- Update `README.md` and `docs/ARCHITECTURE_SPEC.md` when you touch high-level behaviour.
- For major changes, submit an RFC under `docs/rfcs/` before implementation.
- Keep diagrams in Mermaid (preferred) or PlantUML; store rendered images in `docs/img/`.
- Benchmarks belong in `examples/benchmarks/` with reproducible command snippets.

## Pull Requests

- Fill out the PR template (created automatically when you open a PR).
- Provide a short changelog, test evidence, and screenshots/metrics if relevant.
- Request review from the `@qallow-maintainers` team.
- CI must be green before merging. PRs with failing checks will be auto-blocked.

## Code of Conduct

All contributors must act in alignment with the Qallow ethics framework (Sustainability + Compassion + Harmony). Refer to `docs/ETHICS_CHARTER.md` for expectations on behaviour.

## Questions

- Open a GitHub discussion tagged `support`.
- Ping maintainers in the `#qallow-dev` channel of the community workspace.
- For security disclosures, email `security@qallow.ai`.
