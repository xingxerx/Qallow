#!/bin/bash
# Qallow Bend Runner Script
# Provides CLI interface for Bend-based execution

set -e

BEND_DIR="/root/Qallow/bend"
LOG_DIR="/root/Qallow"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_banner() {
    echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║       QALLOW VM - Bend Edition         ║${NC}"
    echo -e "${GREEN}║  Functional Quantum Hardware Emulation ║${NC}"
    echo -e "${GREEN}║  AGI Self-Correction + Error Handling  ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
    echo ""
}

print_usage() {
    echo "Usage: $0 [phase12|phase13|help] <params>"
    echo ""
    echo "Modes:"
    echo "  phase12 <ticks> <epsilon>           - Run elasticity simulation"
    echo "  phase13 <nodes> <ticks> <coupling>  - Run harmonic propagation"
    echo "  help                                 - Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 phase12 100 0.0001"
    echo "  $0 phase13 16 500 0.001"
    echo ""
}

check_bend() {
    if ! command -v bend &> /dev/null; then
        echo -e "${RED}[ERROR]${NC} Bend compiler not found"
        echo "Please install Bend from: https://github.com/HigherOrderCO/Bend"
        exit 1
    fi
    
    local BEND_VERSION=$(bend --version 2>&1 || echo "unknown")
    echo -e "${GREEN}[BEND]${NC} Using Bend: $BEND_VERSION"
}

run_phase12() {
    local TICKS=${1:-100}
    local EPS=${2:-0.0001}
    local LOG_FILE="$LOG_DIR/log_phase12.csv"
    
    echo -e "${YELLOW}[PHASE12]${NC} Running elasticity simulation"
    echo -e "${YELLOW}[PARAMS]${NC} ticks=$TICKS eps=$EPS"
    echo -e "${YELLOW}[OUTPUT]${NC} $LOG_FILE"
    echo ""
    
    cd "$BEND_DIR"
    bend run phase12.bend "$TICKS" "$EPS" > "$LOG_FILE"
    
    echo ""
    echo -e "${GREEN}[SUCCESS]${NC} Simulation complete"
    echo -e "${GREEN}[LOG]${NC} $(wc -l < $LOG_FILE) rows written"
    
    # Show sample output
    echo ""
    echo "Sample output:"
    head -5 "$LOG_FILE"
    echo "..."
    tail -3 "$LOG_FILE"
}

run_phase13() {
    local NODES=${1:-8}
    local TICKS=${2:-400}
    local COUPLING=${3:-0.001}
    local LOG_FILE="$LOG_DIR/log_phase13.csv"
    
    echo -e "${YELLOW}[PHASE13]${NC} Running harmonic propagation"
    echo -e "${YELLOW}[PARAMS]${NC} nodes=$NODES ticks=$TICKS coupling=$COUPLING"
    echo -e "${YELLOW}[OUTPUT]${NC} $LOG_FILE"
    echo ""
    
    cd "$BEND_DIR"
    bend run phase13.bend "$NODES" "$TICKS" "$COUPLING" > "$LOG_FILE"
    
    echo ""
    echo -e "${GREEN}[SUCCESS]${NC} Simulation complete"
    echo -e "${GREEN}[LOG]${NC} $(wc -l < $LOG_FILE) rows written"
    
    # Show sample output
    echo ""
    echo "Sample output:"
    head -5 "$LOG_FILE"
    echo "..."
    tail -3 "$LOG_FILE"
}

# Main execution
print_banner

if [ $# -lt 1 ]; then
    print_usage
    exit 1
fi

MODE=$1
shift

case "$MODE" in
    phase12)
        check_bend
        run_phase12 "$@"
        ;;
    phase13)
        check_bend
        run_phase13 "$@"
        ;;
    help|--help|-h)
        print_usage
        exit 0
        ;;
    *)
        echo -e "${RED}[ERROR]${NC} Unknown mode: $MODE"
        echo ""
        print_usage
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}[QALLOW]${NC} Bend execution completed"
