#!/usr/bin/env bash
# Qallow dependency smoke-checker with optional auto-install helpers.

set -euo pipefail

AUTO_INSTALL=0
EXIT_STATUS=0
APT_UPDATED=0

usage() {
    cat <<'USAGE'
Usage: scripts/check_dependencies.sh [--auto-install]

Checks for required tooling and Python packages. With --auto-install it will
try to install missing items using the detected package manager (apt, dnf,
yum, pacman, or brew) and pip when available.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --auto-install)
            AUTO_INSTALL=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

log_ok()   { printf '✅ %-26s %s\n' "$1" "$2"; }
log_warn() { printf '⚠️  %-26s %s\n' "$1" "$2"; }
log_fail() {
    printf '❌ %-26s %s\n' "$1" "$2";
    EXIT_STATUS=1
}

hint() { printf '   ↳ %s\n' "$1"; }

OS_ID="unknown"
OS_LIKE=""
OS_NAME=$(uname -s)
if [[ -f /etc/os-release ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    OS_ID=${ID:-$OS_ID}
    OS_LIKE=${ID_LIKE:-}
    OS_NAME=${PRETTY_NAME:-$OS_NAME}
fi

PACKAGE_MANAGER=""
if command -v apt-get >/dev/null 2>&1; then
    PACKAGE_MANAGER="apt"
elif command -v dnf >/dev/null 2>&1; then
    PACKAGE_MANAGER="dnf"
elif command -v yum >/dev/null 2>&1; then
    PACKAGE_MANAGER="yum"
elif command -v pacman >/dev/null 2>&1; then
    PACKAGE_MANAGER="pacman"
elif command -v brew >/dev/null 2>&1; then
    PACKAGE_MANAGER="brew"
fi

run_install_cmd() {
    local cmd="$1"
    if [[ $AUTO_INSTALL -eq 0 ]]; then
        return 1
    fi
    echo "📦 Attempting install: $cmd"
    if bash -c "$cmd"; then
        return 0
    fi
    hint "Automatic install failed. Please install manually."
    return 1
}

install_packages() {
    local label="$1"; shift
    local packages=("$@")
    local cmd=""

    case "$PACKAGE_MANAGER" in
        apt)
            if [[ $APT_UPDATED -eq 0 ]]; then
                run_install_cmd "sudo apt-get update" || true
                APT_UPDATED=1
            fi
            cmd="sudo apt-get install -y ${packages[*]}"
            ;;
        dnf)
            cmd="sudo dnf install -y ${packages[*]}"
            ;;
        yum)
            cmd="sudo yum install -y ${packages[*]}"
            ;;
        pacman)
            cmd="sudo pacman -S --noconfirm ${packages[*]}"
            ;;
        brew)
            cmd="brew install ${packages[*]}"
            ;;
        *)
            hint "Automatic install unavailable for ${OS_NAME}."
            return 1
            ;;
    esac

    run_install_cmd "$cmd"
}

ensure_pip_package() {
    local label="$1" package="$2" module="$3"
    if python3 - <<PY >/dev/null 2>&1
import importlib
import sys
try:
    importlib.import_module("$module")
except Exception:
    sys.exit(1)
PY
    then
        local version
        version=$(python3 - <<PY
import importlib
mod = importlib.import_module("$module")
print(getattr(mod, "__version__", "unknown"))
PY
)
        version=${version//$'\n'/}
        log_ok "$label" "version ${version}"
        return 0
    fi

    log_fail "$label" "module not found"
    hint "Install with: python3 -m pip install --user ${package}"
    if [[ $AUTO_INSTALL -eq 1 ]]; then
        if run_install_cmd "python3 -m pip install --user ${package}"; then
            log_ok "$label" "installed"
            return 0
        fi
    fi
    return 1
}

check_python() {
    if ! command -v python3 >/dev/null 2>&1; then
        log_fail "Python >= 3.10" "python3 not found"
        case "$PACKAGE_MANAGER" in
            apt) hint "sudo apt-get install -y python3" ;;
            dnf|yum) hint "sudo $PACKAGE_MANAGER install -y python3" ;;
            pacman) hint "sudo pacman -S python" ;;
            brew) hint "brew install python" ;;
        esac
        install_packages "python3" python3 || true
        return
    fi

    local raw version major minor
    raw=$(python3 --version 2>&1)
    version=${raw#Python }
    IFS='.' read -r major minor _ <<<"$version"
    if (( major > 3 || (major == 3 && minor >= 10) )); then
        log_ok "Python >= 3.10" "$raw"
    else
        log_fail "Python >= 3.10" "$raw"
        hint "Upgrade to Python 3.10+ (pyenv install 3.10.0)"
    fi
}

check_cmake() {
    if ! command -v cmake >/dev/null 2>&1; then
        log_fail "CMake >= 3.20" "cmake not found"
        hint "Install via your package manager"
        install_packages "cmake" cmake || true
        return
    fi
    local raw version major minor
    raw=$(cmake --version | head -n1)
    version=$(cmake --version | head -n1 | awk '{print $3}')
    IFS='.' read -r major minor _ <<<"$version"
    if (( major > 3 || (major == 3 && minor >= 20) )); then
        log_ok "CMake >= 3.20" "$raw"
    else
        log_fail "CMake >= 3.20" "$raw"
        hint "Upgrade CMake to 3.20+"
    fi
}

check_compiler() {
    if command -v gcc >/dev/null 2>&1; then
        log_ok "GCC" "$(gcc --version | head -n1)"
        return
    fi
    if command -v clang >/dev/null 2>&1; then
        log_ok "Clang" "$(clang --version | head -n1)"
        return
    fi
    log_fail "Compiler" "gcc/clang not found"
    hint "Install build-essential or clang"
    install_packages "compilers" build-essential clang || true
}

check_nvcc() {
    if ! command -v nvcc >/dev/null 2>&1; then
        log_warn "CUDA nvcc" "not found"
        hint "Install NVIDIA CUDA Toolkit (optional)"
        return
    fi
    local raw version major
    raw=$(nvcc --version | tail -n1 | sed 's/^ *//')
    version=$(nvcc --version | awk '/release/ {print $6}' | tr -d ',V')
    major=${version%%.*}
    if [[ -z "$major" ]]; then
        log_warn "CUDA nvcc" "$raw"
        return
    fi
    if (( major >= 12 )); then
        log_ok "CUDA nvcc" "$raw"
    else
        log_warn "CUDA nvcc" "$raw"
        hint "Upgrade CUDA toolkit to 12.0+"
    fi
}

check_ncu() {
    if command -v ncu >/dev/null 2>&1; then
        log_ok "Nsight Compute" "$(ncu --version 2>&1 | head -n1)"
        return
    fi
    log_warn "Nsight Compute" "ncu not found"
    hint "Install NVIDIA Nsight Compute (optional)"
}

check_sentence_transformers() {
    ensure_pip_package "sentence-transformers" "sentence-transformers" "sentence_transformers" || true
}

check_cirq() {
    ensure_pip_package "Cirq" "cirq" "cirq" || true
}

check_sdl2() {
    if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists sdl2; then
        log_ok "SDL2" "pkg-config $(pkg-config --modversion sdl2)"
        return
    fi
    if command -v sdl2-config >/dev/null 2>&1; then
        log_ok "SDL2" "sdl2-config $(sdl2-config --version)"
        return
    fi
    log_warn "SDL2" "development headers missing"
    case "$PACKAGE_MANAGER" in
        apt) hint "sudo apt-get install -y libsdl2-dev libsdl2-ttf-dev" ;;
        dnf|yum) hint "sudo $PACKAGE_MANAGER install -y SDL2-devel SDL2_ttf-devel" ;;
        pacman) hint "sudo pacman -S sdl2 sdl2_ttf" ;;
        brew) hint "brew install sdl2 sdl2_ttf" ;;
        *) hint "Install SDL2 development headers" ;;
    esac
    install_packages "SDL2" libsdl2-dev libsdl2-ttf-dev || true
}

check_python
check_cmake
check_compiler
check_nvcc
check_ncu
check_cirq
check_sentence_transformers
check_sdl2

if [[ $EXIT_STATUS -eq 0 ]]; then
    printf '\nAll required dependencies look good on %s.\n' "$OS_NAME"
else
    printf '\nSome dependencies are missing or outdated. See hints above for fixes.\n' >&2
fi

exit $EXIT_STATUS
