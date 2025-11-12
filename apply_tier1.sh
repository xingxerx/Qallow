#!/bin/bash
set -euo pipefail

MANIFEST="cleanup_manifest.json"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST" >&2
  exit 1
fi

echo "Applying Tier 1 (empty shell removals)..."

jq -r '.tiers.tier1_empty_shells[] | select(.action == "delete") | .path' "$MANIFEST" | while read -r dir; do
  [[ -z "$dir" ]] && continue
  if [[ -d "${dir#./}" ]]; then
    echo " - removing $dir"
    rmdir "${dir#./}" 2>/dev/null || echo "   ⚠️  skipped (not empty or protected): $dir"
  else
    echo "   ℹ️  already missing: $dir"
  fi
done

echo "Tier 1 complete. Next step: ./scripts/build_all.sh && ctest"
