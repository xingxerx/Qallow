#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
BUILD_TYPE="RelWithDebInfo"
ENABLE_CUDA="auto"
CLEAN_FIRST=0

usage() {
    cat <<USAGE
Usage: ${0##*/} [--cpu|--cuda|--auto] [--build-type <type>] [--clean]

Options:
  --cpu           Force CPU-only build (disables CUDA)
  --cuda          Force CUDA build (fails if nvcc missing)
  --auto          Auto-detect CUDA (default)
  --build-type    CMake build type (default: RelWithDebInfo)
  --clean         Remove existing build directory before configuring
  --help          Show this message
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu)
            ENABLE_CUDA="OFF"
            ;;
        --cuda)
            ENABLE_CUDA="ON"
            ;;
        --auto)
            ENABLE_CUDA="auto"
            ;;
        --build-type)
            shift
            BUILD_TYPE="${1:-RelWithDebInfo}"
            ;;
        --clean)
            CLEAN_FIRST=1
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
    shift
done

if [[ ${CLEAN_FIRST} -eq 1 ]]; then
    rm -rf "${BUILD_DIR}"
fi

mkdir -p "${BUILD_DIR}"
mkdir -p "${ROOT_DIR}/data/logs"

if [[ "${ENABLE_CUDA}" == "auto" ]]; then
    if command -v nvcc >/dev/null 2>&1; then
        ENABLE_CUDA="ON"
    else
        ENABLE_CUDA="OFF"
    fi
fi

if [[ "${ENABLE_CUDA}" == "ON" ]] && ! command -v nvcc >/dev/null 2>&1; then
    echo "[build_all] nvcc not found but CUDA requested" >&2
    exit 1
fi

echo "[build_all] Configuring (CUDA=${ENABLE_CUDA})"
cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
    -DQALLOW_ENABLE_CUDA="${ENABLE_CUDA}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"

echo "[build_all] Building"
cmake --build "${BUILD_DIR}" --parallel

echo "[build_all] Running tests"
ctest --test-dir "${BUILD_DIR}" --output-on-failure || {
    echo "[build_all] Tests failed" >&2
    exit 1
}

echo "[build_all] Complete"
