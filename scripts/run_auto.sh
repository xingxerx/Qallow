#!/bin/bash
# Helper to launch the phase-13 accelerator binary with sensible defaults.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN=""
cmd_prefix=()
preferred_backend="auto"
enable_qiskit=0
qiskit_backend=""
qiskit_bridge=""

threads="auto"
watch_dir=""
files=()

usage() {
  cat <<EOF
Usage: ${0##*/} [options]

Backend selection:
  --cuda               Force the CUDA binary (requires successful CUDA build)
  --cpu                Force the CPU binary

Qiskit integration:
  --with-qiskit        Export QALLOW_QISKIT=1 for the run
  --qiskit-backend=ID  Set QALLOW_QISKIT_BACKEND to a specific IBM backend
  --qiskit-bridge=PATH Override bridge script (QALLOW_QISKIT_BRIDGE)

Execution:
  --threads=N          Thread count (default: auto)
  --watch=DIR          Watch directory for accelerator mode
  --no-watch           Disable directory watching
  --file=PATH          Provide accelerator input file (repeatable)
  -h, --help           Show this help

Notes:
  - CUDA build can be produced via: ./scripts/build_wrapper.sh CUDA
  - CPU build falls back to ./build/qallow_unified if CUDA binary is unavailable
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda)
      preferred_backend="cuda"
      ;;
    --cpu)
      preferred_backend="cpu"
      ;;
    --with-qiskit)
      enable_qiskit=1
      ;;
    --qiskit-backend=*)
      qiskit_backend="${1#*=}"
      ;;
    --qiskit-backend)
      shift || { echo "[ERROR] Missing value for --qiskit-backend" >&2; usage; exit 1; }
      qiskit_backend="$1"
      ;;
    --qiskit-bridge=*)
      qiskit_bridge="${1#*=}"
      ;;
    --qiskit-bridge)
      shift || { echo "[ERROR] Missing value for --qiskit-bridge" >&2; usage; exit 1; }
      qiskit_bridge="$1"
      ;;
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

select_cuda_binary() {
  local candidates=(
    "${ROOT}/build/qallow_unified_cuda"
    "${ROOT}/build/CUDA/qallow_unified_cuda"
  )
  for candidate in "${candidates[@]}"; do
    if [[ -x "$candidate" ]]; then
      BIN="$candidate"
      cmd_prefix=("run" "--accelerator")
      return 0
    fi
  done
  return 1
}

select_cpu_binary() {
  local candidates=(
    "${ROOT}/build/CPU/qallow_unified_cpu"
    "${ROOT}/build/qallow_unified"
    "${ROOT}/qallow"
  )
  for candidate in "${candidates[@]}"; do
    if [[ -x "$candidate" ]]; then
      BIN="$candidate"
      if [[ "$candidate" == *qallow_unified_cpu || "$candidate" == *qallow_unified ]]; then
        cmd_prefix=("run" "--accelerator")
      else
        cmd_prefix=()
      fi
      return 0
    fi
  done
  return 1
}

case "$preferred_backend" in
  cuda)
    if ! select_cuda_binary; then
      echo "[ERROR] CUDA binary not found. Build it via ./scripts/build_wrapper.sh CUDA" >&2
      exit 1
    fi
    ;;
  cpu)
    if ! select_cpu_binary; then
      echo "[ERROR] CPU binary not found. Build it via make ACCELERATOR=CPU" >&2
      exit 1
    fi
    ;;
  auto)
    if ! select_cuda_binary; then
      if ! select_cpu_binary; then
        echo "[ERROR] Could not find a Qallow executable." >&2
        echo "Build the unified binary via ./scripts/build_wrapper.sh CUDA or make ACCELERATOR=CPU" >&2
        exit 1
      fi
    fi
    ;;
  *)
    echo "[ERROR] Unknown backend preference: $preferred_backend" >&2
    exit 1
    ;;
esac

if (( enable_qiskit )); then
  export QALLOW_QISKIT=1
fi
if [[ -n "$qiskit_backend" ]]; then
  export QALLOW_QISKIT_BACKEND="$qiskit_backend"
fi
if [[ -n "$qiskit_bridge" ]]; then
  export QALLOW_QISKIT_BRIDGE="$qiskit_bridge"
fi

full_cmd=("${cmd_prefix[@]}" "${args[@]}")
echo "[RUN] ${BIN} ${full_cmd[*]}"
exec "${BIN}" "${full_cmd[@]}"
