#!/bin/bash
# Rebuild (AUTO mode) and launch Qallow.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Rebuild with latest sources (auto selects CUDA when available).
"$ROOT/scripts/build_wrapper.sh" AUTO

# Run with accelerator defaults.
"$ROOT/scripts/run_auto.sh" "$@"
