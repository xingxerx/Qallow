#!/bin/bash
# Unified AGI pipeline launcher
#
# This orchestrates the major quantum + classical steps we prepared:
#   1. QSVM Iris workload (GPU by default, falls back to CPU; use --skip-qsvm to disable)
#   2. IBM Quantum Bell-state workload (records job ID; use --skip-ibm to disable)
#   3. Qallow unified binary (with Qiskit bridge enabled by default)
#
# Usage:
#   ./scripts/run_unified_agi.sh [options]
#
# Options:
#   --skip-qsvm            Skip the QSVM Iris classifier step
#   --skip-ibm             Skip the IBM Quantum workload submission
#   --skip-qallow          Skip launching the qallow unified runtime
#   --channel=CHANNEL      IBM Quantum channel (default: ibm_cloud)
#   --backend=NAME         IBM Quantum backend (passes to workloads)
#   --shots=N              Shots for runtime workloads (default: 4000)
#   --resilience=LEVEL     Resilience level for runtime workloads (default: 1)
#   --qallow-args="..."    Extra args passed to run_auto.sh
#   --dry-run              Print the commands without executing them
#
# Prerequisites:
#   - Python virtual environment in ./venv (used for the Python workloads)
#   - Qallow binaries built (CPU or CUDA) so run_auto.sh can locate them
#   - IBM Quantum credentials stored via scripts/ibm_quantum_workload.py or .env

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
VENV_BIN="${ROOT_DIR}/venv/bin"

RUN_QSVM=1
RUN_IBM=1
RUN_QALLOW=1
CHANNEL="ibm_cloud"
BACKEND=""
SHOTS=4000
RESILIENCE=1
QALLOW_EXTRA_ARGS=()
DRY_RUN=0

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

info() {
  echo "[INFO] $*"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-qsvm)
      RUN_QSVM=0
      ;;
    --skip-ibm)
      RUN_IBM=0
      ;;
    --skip-qallow)
      RUN_QALLOW=0
      ;;
    --channel=*)
      CHANNEL="${1#*=}"
      ;;
    --channel)
      shift || die "Missing value for --channel"
      CHANNEL="$1"
      ;;
    --backend=*)
      BACKEND="${1#*=}"
      ;;
    --backend)
      shift || die "Missing value for --backend"
      BACKEND="$1"
      ;;
    --shots=*)
      SHOTS="${1#*=}"
      ;;
    --shots)
      shift || die "Missing value for --shots"
      SHOTS="$1"
      ;;
    --resilience=*)
      RESILIENCE="${1#*=}"
      ;;
    --resilience)
      shift || die "Missing value for --resilience"
      RESILIENCE="$1"
      ;;
    --qallow-args=*)
      QALLOW_EXTRA_ARGS+=("${1#*=}")
      ;;
    --dry-run)
      DRY_RUN=1
      ;;
    --help|-h)
      sed -n '1,80p' "$0"
      exit 0
      ;;
    *)
      die "Unknown option: $1"
      ;;
  esac
  shift
done

if [[ ! -x "${VENV_BIN}/python" ]]; then
  die "Python virtual environment not found at ${VENV_BIN}. Activate or create ./venv first."
fi

export PATH="${VENV_BIN}:${PATH}"

check_python_module() {
  local module="$1"
  if ! "${VENV_BIN}/python" - <<PY >/dev/null 2>&1
import importlib
import sys

try:
    importlib.import_module("${module}")
except Exception:
    sys.exit(1)
PY
  then
    die "Python module '${module}' not installed in ./venv. Install dependencies via: pip install qiskit-aer qiskit-machine-learning scikit-learn"
  fi
}

if (( ! DRY_RUN )); then
  check_python_module "qiskit_aer"
  check_python_module "qiskit_machine_learning"
  check_python_module "sklearn"
fi

run_cmd() {
  if (( DRY_RUN )); then
    echo "[DRY-RUN] $*"
  else
    "$@"
  fi
}

separator() {
  echo "---------------------------------------------------------------------"
}

separator
echo " Qallow Unified AGI Pipeline"
separator

if (( RUN_QSVM )); then
  info "Step 1/3: Running QSVM Iris workload"
  QSVM_CMD=(
    "${VENV_BIN}/python"
    "${SCRIPT_DIR}/qsvm_iris_workload.py"
    "--channel=${CHANNEL}"
    "--shots=${SHOTS}"
    "--resilience-level=${RESILIENCE}"
  )
  if [[ -n "${BACKEND}" ]]; then
    QSVM_CMD+=("--real-backend" "--backend-name" "${BACKEND}")
  fi
  run_cmd "${QSVM_CMD[@]}"
else
  info "Step 1/3: QSVM workload skipped"
fi

separator

if (( RUN_IBM )); then
  info "Step 2/3: Running IBM Quantum Bell-state workload"
  IBM_CMD=(
    "${VENV_BIN}/python"
    "${SCRIPT_DIR}/ibm_quantum_workload.py"
    "--channel=${CHANNEL}"
    "--shots=${SHOTS}"
  )
  if [[ -n "${BACKEND}" ]]; then
    IBM_CMD+=("--backend-name" "${BACKEND}")
  fi
  run_cmd "${IBM_CMD[@]}"
else
  info "Step 2/3: IBM Quantum workload skipped"
fi

separator

if (( RUN_QALLOW )); then
  info "Step 3/3: Launching Qallow unified runtime"
  QALLOW_CMD=("${SCRIPT_DIR}/run_auto.sh" "--with-qiskit")
  if [[ -n "${BACKEND}" ]]; then
    QALLOW_CMD+=("--qiskit-backend=${BACKEND}")
  fi
  if [[ ${#QALLOW_EXTRA_ARGS[@]} -gt 0 ]]; then
    QALLOW_CMD+=("${QALLOW_EXTRA_ARGS[@]}")
  fi
  run_cmd "${QALLOW_CMD[@]}"
else
  info "Step 3/3: Qallow runtime launch skipped"
fi

separator
echo " Unified AGI pipeline complete."
if (( DRY_RUN )); then
  echo " (commands were not executed due to --dry-run)"
fi
separator

echo "If the Qallow runtime greets you, it should appear in its stdout."
echo "Logs and metrics remain under data/logs/ by default."
