#!/usr/bin/env bash
set -euo pipefail

PREFIX=${1:-/etc/qallow}
LOG_DIR=${2:-/var/log/qallow}

info() {
  printf '[phase16] %s
' "$1"
}

warn() {
  printf '[phase16][warn] %s
' "$1" >&2
}

# Create objective map template
if mkdir -p "$PREFIX" 2>/dev/null; then
  info "ensured objective directory: $PREFIX"
else
  warn "unable to create $PREFIX (insufficient permissions?)"
fi

OBJECTIVES_FILE="$PREFIX/objectives.json"
if [ ! -f "$OBJECTIVES_FILE" ]; then
  cat >"$OBJECTIVES_FILE" <<'JSON'
{
  "phase7.goal_commit": "Phase 7 governance goals committed",
  "phase7.plan_eval": "Transfer engine plan evaluation",
  "phase7.reflection": "Self-reflection confidence sweep",
  "phase12.elasticity": "Maintain elasticity coherence",
  "phase13.harmonic": "Sustain harmonic alignment"
}
JSON
  info "wrote objective map: $OBJECTIVES_FILE"
else
  info "objective map already present: $OBJECTIVES_FILE"
fi

# Ensure log directories exist
if mkdir -p "$LOG_DIR" "$LOG_DIR/archive" 2>/dev/null; then
  info "ensured log directories under $LOG_DIR"
else
  warn "unable to create $LOG_DIR (insufficient permissions?)"
fi

info "phase16 meta-introspect resources registered"
