#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=false
if [[ ${1-} == "--dry-run" ]]; then
  DRY_RUN=true
fi

payload=$(cat)
if [[ -z "$payload" ]]; then
  echo "[ERROR] No payload received on stdin" >&2
  exit 1
fi

pod_name=$(python3 - <<'PY'
import json, sys
payload = sys.stdin.read()
try:
    data = json.loads(payload)
except json.JSONDecodeError:
    print("")
    sys.exit(0)
alerts = data.get("alerts", [])
if not alerts:
    print("")
    sys.exit(0)
labels = alerts[0].get("labels", {})
print(labels.get("pod", ""))
PY <<<"$payload")

if [[ -z "$pod_name" ]]; then
  echo "[WARN] Unable to determine offending pod from payload" >&2
  exit 0
fi

echo "[INFO] Auto-halt triggered by alert for pod: $pod_name" >&2
if [[ "$DRY_RUN" == "true" ]]; then
  echo "[DRY-RUN] kubectl scale deployment/qallow-core --replicas=0" >&2
  exit 0
fi

if ! command -v kubectl >/dev/null 2>&1; then
  echo "[ERROR] kubectl not available in PATH" >&2
  exit 1
fi

kubectl scale deployment/qallow-core --replicas=0 >/dev/null
echo "[INFO] qallow-core scaled to 0 replicas" >&2
