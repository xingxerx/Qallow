#!/bin/bash
set -euo pipefail

MANIFEST="cleanup_manifest.json"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST" >&2
  exit 1
fi

echo "Applying Tier 2 (duplicate symlink replacements)..."

jq -r '.tiers.tier2_duplicate_symlinks[] | "\(.from)|\(.to)"' "$MANIFEST" | while IFS='|' read -r src dst; do
  [[ -z "$src" || -z "$dst" ]] && continue
  src_path="${src#./}"
  dst_path="${dst#./}"
  if [[ ! -f "$dst_path" ]]; then
    echo "   ⚠️  canonical target missing, skipping: $src_path -> $dst_path"
    continue
  fi
  if [[ -e "$src_path" ]]; then
    echo " - replacing $src_path with symlink to $dst_path"
    rm -f "$src_path"
  else
    echo "   ℹ️  $src_path absent, will create symlink"
  fi
  rel_target=$(python3 - "$src_path" "$dst_path" <<'PY'
import os, sys
src, dst = sys.argv[1:3]
base = os.path.dirname(src) or '.'
print(os.path.relpath(dst, base))
PY
)
  ln -s "$rel_target" "$src_path"
done

echo "Tier 2 complete. Rebuild to verify behavior."
