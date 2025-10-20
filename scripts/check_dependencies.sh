#!/usr/bin/env bash
# Qallow dependency smoke-checker for internal readiness gates.

set -euo pipefail

STATUS=0

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

# Python version check (>= 3.13)
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
    if [[ "$PY_MAJOR" -ge 3 && "$PY_MINOR" -ge 13 ]]; then
        print_status 0 "Python >= 3.13" "$PY_RAW"
    else
        print_status 1 "Python >= 3.13" "$PY_RAW"
    fi
else
    print_status 1 "Python >= 3.13" "python3 not found"
fi

# CUDA toolkit check (nvcc >= 13.0)
if command -v nvcc >/dev/null 2>&1; then
    NVCC_RAW=$(nvcc --version | tail -n 1 | sed 's/^ *//')
    NVCC_VER=$(nvcc --version | awk '/release/ {print $6}' | tr -d ',V')
    if [[ "$NVCC_VER" =~ ^([0-9]+)\.([0-9]+) ]]; then
        MAJOR="${BASH_REMATCH[1]}"
        if (( MAJOR >= 13 )); then
            print_status 0 "CUDA nvcc >= 13.0" "$NVCC_RAW"
        else
            print_status 1 "CUDA nvcc >= 13.0" "$NVCC_RAW"
        fi
    else
        print_status 1 "CUDA nvcc >= 13.0" "Unable to parse version from: $NVCC_RAW"
    fi
else
    print_status 1 "CUDA nvcc >= 13.0" "nvcc not found"
fi

# Nsight Compute CLI presence
if command -v nv-nsight-cu-cli >/dev/null 2>&1; then
    NSIGHT_BIN="$(command -v nv-nsight-cu-cli)"
    NSIGHT_VER=$("$NSIGHT_BIN" --version 2>&1 | head -n 1)
    print_status 0 "nv-nsight-cu-cli" "$NSIGHT_VER"
elif [[ -x /opt/cuda/NsightCompute/nv-nsight-cu-cli ]]; then
    NSIGHT_BIN="/opt/cuda/NsightCompute/nv-nsight-cu-cli"
    NSIGHT_VER=$("$NSIGHT_BIN" --version 2>&1 | head -n 1)
    print_status 0 "nv-nsight-cu-cli" "$NSIGHT_VER"
else
    print_status 1 "nv-nsight-cu-cli" "binary not found in PATH or /opt/cuda/NsightCompute"
fi

# SentenceTransformer model availability
PY_CHECK=$(python3 - <<'PY'
import json
res = {"import": False, "loaded": False, "error": None}
try:
    from sentence_transformers import SentenceTransformer
    res["import"] = True
    try:
        SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        res["loaded"] = True
    except Exception as exc:
        res["error"] = str(exc)
except Exception as exc:
    res["error"] = str(exc)
print(json.dumps(res))
PY
 2>/dev/null || true)

if [[ -z "$PY_CHECK" ]]; then
    print_status 1 "sentence-transformers model" "Python execution failed"
else
    IMPORT_OK=$(python3 - <<'PY' "$PY_CHECK"
import json, sys
data = json.loads(sys.argv[1])
print("1" if data["import"] else "0")
PY
    )
    LOADED_OK=$(python3 - <<'PY' "$PY_CHECK"
import json, sys
data = json.loads(sys.argv[1])
print("1" if data["loaded"] else "0")
PY
    )
    ERROR_MSG=$(python3 - <<'PY' "$PY_CHECK"
import json, sys
data = json.loads(sys.argv[1])
print(data["error"] or "")
PY
    )
    if [[ "$IMPORT_OK" == "1" && "$LOADED_OK" == "1" ]]; then
        print_status 0 "sentence-transformers model" "all-MiniLM-L6-v2 available"
    else
        DETAILS="missing dependency"
        if [[ -n "$ERROR_MSG" ]]; then
            DETAILS="$ERROR_MSG"
        fi
        print_status 1 "sentence-transformers model" "$DETAILS"
    fi
fi

exit $STATUS
