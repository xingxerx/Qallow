#!/bin/bash
# Rebuild (AUTO mode) and launch Qallow with Qiskit integration.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Rebuild with latest sources (auto selects CUDA when available).
"$ROOT/scripts/build_wrapper.sh" AUTO

# Run with accelerator defaults and Qiskit enabled unless overridden.
"$ROOT/scripts/run_auto.sh" --with-qiskit "$@"
