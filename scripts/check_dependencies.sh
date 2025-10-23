#!/usr/bin/env bash
# Qallow dependency smoke-checker for internal readiness gates.
# Enhanced with auto-install capability

set -euo pipefail

STATUS=0
AUTO_INSTALL=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --auto-install)
            AUTO_INSTALL=1
            shift
            ;;
        *)
            shift
            ;;
    esac
done

print_status() {
    local ok="$1"
    local label="$2"
    local details="$3"
    if [[ "$ok" == "0" ]]; then
        printf "✅ %-28s %s\n" "$label" "$details"
    else
        printf "⚠️  %-28s %s\n" "$label" "$details"
        STATUS=1
    fi
}

install_package() {
    local package="$1"
    local installer="$2"

    if [[ $AUTO_INSTALL -eq 1 ]]; then
        printf "📦 Installing %s...\n" "$package"
        eval "$installer" || return 1
        return 0
    fi
    return 1
}

# Python version check (>= 3.10)
if command -v python3 >/dev/null 2>&1; then
    PY_RAW=$(python3 --version 2>&1)
    PY_MAJOR=$(python3 - <<'PY'
import sys
print(sys.version_info.major)
PY
)
    PY_MINOR=$(python3 - <<'PY'
import sys
print(sys.version_info.minor)
PY
)
    if [[ "$PY_MAJOR" -ge 3 && "$PY_MINOR" -ge 10 ]]; then
        print_status 0 "Python >= 3.10" "$PY_RAW"
    else
        print_status 1 "Python >= 3.10" "$PY_RAW"
    fi
else
    print_status 1 "Python >= 3.10" "python3 not found"
    if install_package "python3" "apt-get install -y python3"; then
        print_status 0 "Python >= 3.10" "Installed successfully"
    fi
fi

# CUDA toolkit check (nvcc >= 13.0)
if command -v nvcc >/dev/null 2>&1; then
    NVCC_RAW=$(nvcc --version | tail -n 1 | sed 's/^ *//')
    NVCC_VER=$(nvcc --version | awk '/release/ {print $6}' | tr -d ',V')
    if [[ "$NVCC_VER" =~ ^([0-9]+)\.([0-9]+) ]]; then
        MAJOR="${BASH_REMATCH[1]}"
        if (( MAJOR >= 13 )); then
            #!/usr/bin/env bash
            # Qallow dependency smoke-checker for internal readiness gates.

            set -euo pipefail

            readonly CHECK_LABEL_WIDTH=28

            log_ok() {
                printf "✅ %-*s %s\n" "$CHECK_LABEL_WIDTH" "$1" "$2"
            }

            log_fail() {
                printf "❌ %-*s %s\n" "$CHECK_LABEL_WIDTH" "$1" "$2"
            }

            hint() {
                printf "   ↳ %s\n" "$1"
            }

            detect_os() {
                local id_like="" id="" pretty=""
                if [[ -f /etc/os-release ]]; then
                    # shellcheck disable=SC1091
                    source /etc/os-release
                    id=${ID:-}
                    id_like=${ID_LIKE:-}
                    pretty=${PRETTY_NAME:-}
                elif [[ "$(uname -s)" == "Darwin" ]]; then
                    id="darwin"
                    pretty="macOS"
                else
                    id="unknown"
                    pretty=$(uname -s)
                fi
                printf '%s;%s;%s' "$id" "$id_like" "$pretty"
            }

            fail_with_hints() {
                local label="$1"; shift
                log_fail "$label" "missing"
                for msg in "$@"; do
                    hint "$msg"
                done
                exit 1
            }

            OS_INFO=$(detect_os)
            OS_ID=${OS_INFO%%;*}
            OS_REST=${OS_INFO#*;}
            OS_LIKE=${OS_REST%%;*}
            OS_NAME=${OS_REST#*;}

            python_check() {
                if ! command -v python3 >/dev/null 2>&1; then
                    fail_with_hints "Python >= 3.13" \
                        "Install Python 3.13+ (e.g. pyenv, asdf, system package)"
                fi

                local raw
                raw=$(python3 --version 2>&1)
                local major minor
                IFS='.' read -r major minor _ <<<"${raw#Python }"
                if (( major < 3 || minor < 13 )); then
                    fail_with_hints "Python >= 3.13" "Detected ${raw}. Upgrade to 3.13+ (pyenv install 3.13.0)."
                fi
                log_ok "Python >= 3.13" "$raw"
            }

            cuda_check() {
                if ! command -v nvcc >/dev/null 2>&1; then
                    fail_with_hints "CUDA nvcc >= 12.0" \
                        "nvcc not found on PATH" \
                        "Ubuntu: sudo apt-get install cuda-toolkit-12-4" \
                        "Arch: sudo pacman -S cuda" \
                        "Refer to NVIDIA CUDA installation guide"
                fi

                local raw version major
                raw=$(nvcc --version | tail -n1 | sed 's/^ *//')
                version=$(nvcc --version | awk '/release/ {print $6}' | tr -d ',V')
                if [[ $version =~ ^([0-9]+)\.([0-9]+) ]]; then
                    major=${BASH_REMATCH[1]}
                    if (( major < 12 )); then
                        fail_with_hints "CUDA nvcc >= 12.0" "Detected $raw"
                    fi
                    log_ok "CUDA nvcc >= 12.0" "$raw"
                else
                    fail_with_hints "CUDA nvcc >= 12.0" "Unable to parse nvcc version from: $raw"
                fi
            }

            find_ncu_candidate() {
                if command -v ncu >/dev/null 2>&1; then
                    command -v ncu
                    return
                fi

                local probes=("${CUDA_PATH:-}/NsightCompute/ncu" "${CUDA_PATH:-}/bin/ncu" \
                              "/usr/local/cuda/bin/ncu" "/usr/local/cuda/NsightCompute/ncu" \
                              "/opt/cuda/NsightCompute/ncu")
                for candidate in "${probes[@]}"; do
                    if [[ -x "$candidate" ]]; then
                        printf '%s' "$candidate"
                        return
                    fi
                done
                return 1
            }

            ncu_check() {
                local ncu_bin
                if ! ncu_bin=$(find_ncu_candidate); then
                    if [[ $OS_ID == "arch" || $OS_LIKE == *"arch"* ]]; then
                        fail_with_hints "Nsight Compute (ncu)" \
                            "ncu not found" \
                            "Download the standalone installer from NVIDIA Developer (Nsight Compute)" \
                            "Arch note: there is no official pacman package; use the installer or trusted AUR" \
                            "After install ensure ncu --version succeeds"
                    fi
                    fail_with_hints "Nsight Compute (ncu)" \
                        "ncu not found" \
                        "Use the NVIDIA Nsight Compute installer or CUDA toolkit >= 12.0" \
                        "Preferred command: ncu (nv-nsight-cu-cli is deprecated)"
                fi

                local version
                version=$("$ncu_bin" --version 2>&1 | head -n1)
                log_ok "Nsight Compute" "$version (${ncu_bin})"
            }

            python_import_check() {
                local label="$1" module="$2" hint_msg="$3"
                local script="import importlib, sys; mod = importlib.import_module('${module}')\nprint(getattr(mod, '__version__', 'unknown'))"
                local output
                if ! output=$(python3 - <<PY 2>/dev/null
            $script
            PY
            ); then
                    fail_with_hints "$label" "$module import failed" "$hint_msg"
                fi
                log_ok "$label" "version ${output//\n/}"
            }

            sentence_transformers_check() {
                local result
                if ! result=$(python3 - <<'PY'
            import json
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            print(json.dumps({"device": str(model.device)}))
            PY
            ); then
                    fail_with_hints "sentence-transformers" \
                        "Model load failed" \
                        "Run: python3 -m pip install -r requirements.txt" \
                        "Ensure GPUs have internet access for first-time model download"
                fi
                log_ok "sentence-transformers" "all-MiniLM-L6-v2 ready ${result//\n/}"
            }

            sdl_check() {
                if command -v pkg-config >/dev/null 2>&1; then
                    if pkg-config --exists sdl2; then
                        local version
                        version=$(pkg-config --modversion sdl2)
                        log_ok "SDL2" "pkg-config ${version}"
                        return
                    fi
                fi
                if command -v sdl2-config >/dev/null 2>&1; then
                    local version
                    version=$(sdl2-config --version)
                    log_ok "SDL2" "sdl2-config ${version}"
                    return
                fi

                fail_with_hints "SDL2" \
                    "Development headers not found" \
                    "Ubuntu: sudo apt-get install libsdl2-dev libsdl2-ttf-dev" \
                    "Arch: sudo pacman -S sdl2 sdl2_ttf" \
                    "macOS (Homebrew): brew install sdl2 sdl2_ttf"
            }

            python_check
            cuda_check
            ncu_check
            python_import_check "Qiskit" "qiskit" "Run: python3 -m pip install -r requirements.txt"
            sentence_transformers_check
            sdl_check

            printf '\nAll critical dependencies satisfied on %s.\n' "$OS_NAME"
