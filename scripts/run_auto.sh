#!/bin/bash
# Helper to launch the phase-13 accelerator binary with sensible defaults.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN=""
cmd_prefix=()

if [[ -x "${ROOT}/build/qallow_unified" ]]; then
  BIN="${ROOT}/build/qallow_unified"
  cmd_prefix=("run" "--accelerator")
elif [[ -x "${ROOT}/qallow" ]]; then
  BIN="${ROOT}/qallow"
  cmd_prefix=()
else
  echo "[ERROR] Could not find a Qallow executable." >&2
  echo "Build the unified binary via ./scripts/build_wrapper.sh CUDA" >&2
  echo "or compile the standalone accelerator with:" >&2
  echo "  gcc -O3 -march=native -flto -pthread src/qallow_phase13.c -o qallow" >&2
  exit 1
fi

threads="auto"
watch_dir=""
files=()

usage() {
  cat <<EOF
Usage: ${0##*/} [--threads=N] [--watch=DIR] [--file=PATH]...
If no watch or file is provided, the current directory is watched.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --threads=*)
      threads="${1#*=}"
      ;;
    --threads)
      shift || { echo "[ERROR] Missing value for --threads" >&2; usage; exit 1; }
      threads="$1"
      ;;
    --watch=*)
      watch_dir="${1#*=}"
      ;;
    --watch)
      shift || { echo "[ERROR] Missing value for --watch" >&2; usage; exit 1; }
      watch_dir="$1"
      ;;
    --no-watch)
      watch_dir=""
      ;;
    --file=*)
      files+=("${1#*=}")
      ;;
    --file)
      shift || { echo "[ERROR] Missing value for --file" >&2; usage; exit 1; }
      files+=("$1")
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      files+=("$1")
      ;;
  esac
  shift
done

args=("--threads=${threads}")
if [[ -n "$watch_dir" ]]; then
  args+=("--watch=${watch_dir}")
fi

if [[ -z "$watch_dir" && ${#files[@]} -eq 0 ]]; then
  args+=("--watch=$(pwd)")
fi

for f in "${files[@]}"; do
  args+=("--file=${f}")
done

full_cmd=("${cmd_prefix[@]}" "${args[@]}")
echo "[RUN] ${BIN} ${full_cmd[*]}"
exec "${BIN}" "${full_cmd[@]}"
