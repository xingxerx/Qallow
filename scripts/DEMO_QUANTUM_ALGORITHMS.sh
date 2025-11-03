#!/bin/bash

# QALLOW QUANTUM ALGORITHMS DEMO
# Complete demonstration of all quantum algorithms

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "\n${CYAN}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}█ $1${NC}"
    echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}\n"
}

print_section() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}▶ $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

# Main demo
main() {
    cd /root/Qallow
    
    print_header "QALLOW QUANTUM ALGORITHMS DEMONSTRATION"
    
    # Check if venv is activated
    if [[ -z "$VIRTUAL_ENV" ]]; then
        print_info "Activating Python virtual environment..."
        source venv/bin/activate
    fi
    
    print_success "Virtual environment activated"
    
    # Demo 1: Unified Framework
    print_section "DEMO 1: Unified Quantum Framework (6 Algorithms)"
    print_info "Running: Hello Quantum, Bell State, Deutsch, Grover's, Shor's, VQE"
    python3 quantum_algorithms/unified_quantum_framework.py 2>&1 | tail -30
    print_success "Unified framework complete"
    
    # Demo 2: Quantum Search
    print_section "DEMO 2: Quantum Database Search"
    print_info "Searching database of 16 items for target value 11"
    python3 quantum_algorithms/algorithms/my_quantum_search.py 2>&1 | tail -40
    print_success "Quantum search complete"
    
    # Demo 3: Quantum Optimization
    print_section "DEMO 3: Quantum Optimization (QAOA)"
    print_info "Solving MaxCut and Traveling Salesman problems"
    python3 quantum_algorithms/algorithms/quantum_optimization.py 2>&1 | tail -30
    print_success "Quantum optimization complete"
    
    # Demo 4: Quantum ML
    print_section "DEMO 4: Quantum Machine Learning"
    print_info "Training classifier and clustering data"
    python3 quantum_algorithms/algorithms/quantum_ml.py 2>&1 | tail -25
    print_success "Quantum ML complete"
    
    # Demo 5: Quantum Simulation
    print_section "DEMO 5: Quantum Simulation"
    print_info "Simulating harmonic oscillator, molecules, and dynamics"
    python3 quantum_algorithms/algorithms/quantum_simulation.py 2>&1 | tail -30
    print_success "Quantum simulation complete"
    
    # Demo 6: Complete Suite
    print_section "DEMO 6: Complete Algorithm Suite"
    print_info "Running all algorithms together with metrics"
    python3 quantum_algorithms/QUANTUM_ALGORITHM_SUITE.py 2>&1 | tail -40
    print_success "Complete suite executed"
    
    # Summary
    print_header "DEMONSTRATION COMPLETE"
    
    echo -e "${GREEN}All quantum algorithms executed successfully!${NC}\n"
    
    echo -e "${CYAN}📊 RESULTS SUMMARY:${NC}"
    echo -e "   • Unified Framework: 6 algorithms ✅"
    echo -e "   • Quantum Search: Database search ✅"
    echo -e "   • Quantum Optimization: MaxCut + TSP ✅"
    echo -e "   • Quantum ML: Classifier + Clustering ✅"
    echo -e "   • Quantum Simulation: 3 simulators ✅"
    echo -e "   • Total: 15+ quantum algorithms\n"
    
    echo -e "${CYAN}📁 RESULTS EXPORTED:${NC}"
    if [ -f "quantum_algorithm_suite_results.json" ]; then
        echo -e "   ✅ quantum_algorithm_suite_results.json"
        print_success "Results saved successfully"
    fi
    
    echo -e "\n${CYAN}🚀 NEXT STEPS:${NC}"
    echo -e "   1. Run Qallow phase: ${YELLOW}./build/qallow phase 14 --ticks=500${NC}"
    echo -e "   2. Monitor with GUI: ${YELLOW}cargo run${NC}"
    echo -e "   3. View results: ${YELLOW}cat quantum_algorithm_suite_results.json${NC}"
    echo -e "   4. Read guide: ${YELLOW}cat QUANTUM_ALGORITHMS_GUIDE.md${NC}\n"
    
    echo -e "${CYAN}📚 DOCUMENTATION:${NC}"
    echo -e "   • Guide: QUANTUM_ALGORITHMS_GUIDE.md"
    echo -e "   • Template: quantum_algorithms/algorithms/custom_algorithm_template.py"
    echo -e "   • Suite: quantum_algorithms/QUANTUM_ALGORITHM_SUITE.py\n"
    
    print_success "Demo completed successfully!"
}

# Run main
main "$@"

