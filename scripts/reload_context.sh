#!/usr/bin/env bash

# reload_context.sh - convenience script to print persistent notes

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NOTES_FILE="$ROOT_DIR/docs/PERSISTENT_NOTES.md"

if [[ ! -f "$NOTES_FILE" ]]; then
  echo "[ERROR] Persistent notes file not found at $NOTES_FILE" >&2
  exit 1
fi

echo "[INFO] Copy the following context into the new session:\n"
cat "$NOTES_FILE"
