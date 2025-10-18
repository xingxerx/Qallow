#!/bin/bash
# Qallow Bend Emulation Runner
# Runs Bend simulations even without Bend compiler installed
# Uses Python to simulate Bend's functional execution

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BEND_DIR="$SCRIPT_DIR/../bend"
LOG_DIR="$SCRIPT_DIR/.."

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_banner() {
    echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║       QALLOW VM - Bend Edition         ║${NC}"
    echo -e "${GREEN}║  Functional Quantum Hardware Emulation ║${NC}"
    echo -e "${GREEN}║  AGI Self-Correction + Error Handling  ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
    echo ""
}

# Python emulator for Bend Phase 12
run_phase12_emulated() {
    local TICKS=$1
    local EPS=$2
    local LOG_FILE=$3
    
    python3 << EOF
import sys
import math

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def step(state, eps):
    entropy = clamp(state['entropy'] - 0.000001 + eps * 0.0000002, 0.0, 0.001)
    coherence = clamp(1.0 - entropy * 0.2, 0.0, 1.0)
    deco = clamp(state['deco'] * (1.0 - 0.0005) + eps * 0.0000001, 0.0, 0.001)
    return {'coherence': coherence, 'entropy': entropy, 'deco': deco}

def audit(value):
    """AGI self-correction: clamp values to valid range"""
    if value < 0.0 or value > 1.0:
        print(f"[AUDIT] ⚠️  Value {value:.6f} out of range, clamping", file=sys.stderr)
        return clamp(value, 0.0, 1.0)
    return value

def phase12(ticks, eps):
    state = {'coherence': 0.99990, 'entropy': 0.00070, 'deco': 0.000009}
    results = []
    
    for t in range(1, ticks + 1):
        state = step(state, eps)
        # Apply audit to each value
        coherence = audit(state['coherence'])
        entropy = audit(state['entropy'])
        deco = audit(state['deco'])
        results.append((t, coherence, entropy, deco))
    
    return results

# Run simulation
ticks = $TICKS
eps = $EPS

print("tick,coherence,entropy,decoherence")
rows = phase12(ticks, eps)
for row in rows:
    print(f"{row[0]},{row[1]:.6f},{row[2]:.6f},{row[3]:.6f}")

print(f"[PHASE12] Completed: ticks={ticks} eps={eps:.6f}", file=sys.stderr)
print(f"[AGI] Final coherence≈{rows[-1][1]:.6f}", file=sys.stderr)
EOF
}

# Python emulator for Bend Phase 13
run_phase13_emulated() {
    local NODES=$1
    local TICKS=$2
    local COUPLING=$3
    local LOG_FILE=$4
    
    python3 << EOF
import sys
import math

def init_phases(n):
    return [2.0 * math.pi * i / n for i in range(n)]

def avg(xs):
    return sum(xs) / len(xs)

def step(phases, k):
    m = avg(phases)
    return [p + k * (m - p) for p in phases]

def drift(phases):
    m = avg(phases)
    return avg([abs(m - p) for p in phases])

def audit(value):
    """AGI self-correction: clamp values to valid range"""
    if value < 0.0 or value > 1.0:
        print(f"[AUDIT] ⚠️  Value {value:.6f} out of range, clamping", file=sys.stderr)
        return max(0.0, min(1.0, value))
    return value

def phase13(nodes, ticks, k):
    phases = init_phases(nodes)
    results = []
    
    for t in range(1, ticks + 1):
        d = drift(phases)
        coh = 1.0 / (1.0 + d * 1000.0)
        # Apply audit
        coh = audit(coh)
        d = audit(d)
        results.append((t, coh, d))
        phases = step(phases, k)
    
    return results

# Run simulation
nodes = $NODES
ticks = $TICKS
coupling = $COUPLING

print("tick,avg_coherence,phase_drift")
rows = phase13(nodes, ticks, coupling)
for row in rows:
    print(f"{row[0]},{row[1]:.6f},{row[2]:.6f}")

print(f"[PHASE13] Completed: nodes={nodes} ticks={ticks} k={coupling:.6f}", file=sys.stderr)
print(f"[AGI] Final coherence≈{rows[-1][1]:.6f}", file=sys.stderr)
EOF
}

# Main execution
MODE=$1
shift

case "$MODE" in
    phase12)
        TICKS=${1:-100}
        EPS=${2:-0.0001}
        LOG_FILE="$LOG_DIR/log_phase12.csv"
        
        echo -e "${YELLOW}[PHASE12]${NC} Elasticity Simulation (Bend Emulated)"
        echo -e "${YELLOW}[PARAMS]${NC} ticks=$TICKS eps=$EPS"
        echo -e "${YELLOW}[OUTPUT]${NC} $LOG_FILE"
        echo ""
        
        run_phase12_emulated "$TICKS" "$EPS" "$LOG_FILE" > "$LOG_FILE" 2>&1
        
        echo ""
        echo -e "${GREEN}[SUCCESS]${NC} $(grep -c "^[0-9]" $LOG_FILE || echo 0) data rows written"
        echo -e "${GREEN}[SAMPLE]${NC} First 5 rows:"
        grep "^tick," "$LOG_FILE" 2>/dev/null || true
        grep "^[0-9]" "$LOG_FILE" 2>/dev/null | head -5 || true
        ;;
        
    phase13)
        NODES=${1:-8}
        TICKS=${2:-400}
        COUPLING=${3:-0.001}
        LOG_FILE="$LOG_DIR/log_phase13.csv"
        
        echo -e "${YELLOW}[PHASE13]${NC} Harmonic Propagation (Bend Emulated)"
        echo -e "${YELLOW}[PARAMS]${NC} nodes=$NODES ticks=$TICKS k=$COUPLING"
        echo -e "${YELLOW}[OUTPUT]${NC} $LOG_FILE"
        echo ""
        
        run_phase13_emulated "$NODES" "$TICKS" "$COUPLING" "$LOG_FILE" > "$LOG_FILE" 2>&1
        
        echo ""
        echo -e "${GREEN}[SUCCESS]${NC} $(grep -c "^[0-9]" $LOG_FILE || echo 0) data rows written"
        echo -e "${GREEN}[SAMPLE]${NC} First 5 rows:"
        grep "^tick," "$LOG_FILE" 2>/dev/null || true
        grep "^[0-9]" "$LOG_FILE" 2>/dev/null | head -5 || true
        ;;
        
    *)
        echo -e "${RED}[ERROR]${NC} Unknown mode: $MODE"
        echo "Usage: $0 [phase12|phase13] <params>"
        exit 1
        ;;
esac

echo ""
echo -e "${BLUE}[INFO]${NC} Using Python emulation (native Bend not installed)"
echo -e "${BLUE}[INFO]${NC} Install Bend for optimal performance: https://github.com/HigherOrderCO/Bend"
