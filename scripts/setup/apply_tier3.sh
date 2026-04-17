#!/bin/bash
set -euo pipefail

MANIFEST="cleanup_manifest.json"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST" >&2
  exit 1
fi

echo "Tier 3 is review-only. This script prints candidates for manual action."
echo
jq -r '.tiers.tier3_stale_review[] | "\(.risk)\t\(.path)\t\(.reason)"' "$MANIFEST" | head -n 200
echo
echo "Review each path, update the manifest with APPROVED entries, then craft targeted patches."
