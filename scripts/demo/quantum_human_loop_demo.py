#!/usr/bin/env python3
"""
QHIL Demo: Automated demonstration of Quantum Human-in-the-Loop algorithm
Shows how the system works without requiring interactive input
"""

import sys
sys.path.insert(0, '/home/xing/Qallow')

from quantum_human_loop import (
    QuantumHumanInTheLoopOptimizer,
    QuantumHumanInteractionLanguage,
    QuantumState
)
import numpy as np
import json
from datetime import datetime

def demo_qhil():
    """Run automated QHIL demonstration"""
    print("\n" + "="*80)
    print("QUANTUM HUMAN-IN-THE-LOOP (QHIL) ALGORITHM - AUTOMATED DEMO")
    print("="*80)
    
    # Initialize optimizer
    optimizer = QuantumHumanInTheLoopOptimizer(n_qubits=3, max_depth=2)
    
    print("\n[PHASE 1] Quantum Circuit Initialization")
    print("-" * 80)
    print(f"✓ Initialized {optimizer.n_qubits} qubits")
    print(f"✓ Max circuit depth: {optimizer.max_depth}")
    print(f"✓ Simulator: Cirq (v1.6.1)")
    
    # Simulate human feedback sequence
    feedback_sequence = [
        "DEEPEN STRONG",      # Increase circuit depth
        "ENTANGLE MEDIUM",    # Amplify entanglement
        "PHASE WEAK",         # Rotate phases slightly
        "DENOISE STRONG",     # Reduce noise
        "ACCEPT",             # Accept final state
    ]
    
    print("\n[PHASE 2] Interactive Optimization Loop")
    print("-" * 80)
    
    depth = 2
    params = np.random.randn(optimizer.n_qubits * 2 * depth) * 0.1
    
    for iteration, feedback in enumerate(feedback_sequence):
        print(f"\n[Iteration {iteration + 1}] Human Feedback: '{feedback}'")
        
        # Execute quantum step
        state = optimizer.step(depth, params)
        
        # Display quantum state
        msg = QuantumHumanInteractionLanguage.encode_state(state)
        print(f"  Quantum State: {msg}")
        print(f"  Fidelity: {state.fidelity:.6f}")
        print(f"  Entropy: {state.entropy:.6f}")
        
        # Decode and apply feedback
        decoded = QuantumHumanInteractionLanguage.decode_feedback(feedback)
        print(f"  Decoded Command: {decoded['command']}")
        print(f"  Intensity: {decoded['intensity']:.2f}")
        
        # Apply feedback to parameters
        if decoded['command'] != 'accept':
            params = optimizer.apply_human_feedback(feedback, params)
            print(f"  ✓ Parameters updated")
        else:
            print(f"  ✓ State accepted")
    
    print("\n[PHASE 3] Results Analysis")
    print("-" * 80)
    
    history = optimizer.history
    print(f"Total iterations: {len(history)}")
    print(f"Final fidelity: {history[-1].fidelity:.6f}")
    print(f"Final entropy: {history[-1].entropy:.6f}")
    
    # Compute statistics
    fidelities = [s.fidelity for s in history]
    entropies = [s.entropy for s in history]
    
    print(f"\nFidelity Statistics:")
    print(f"  Min: {min(fidelities):.6f}")
    print(f"  Max: {max(fidelities):.6f}")
    print(f"  Mean: {np.mean(fidelities):.6f}")
    print(f"  Std: {np.std(fidelities):.6f}")
    
    print(f"\nEntropy Statistics:")
    print(f"  Min: {min(entropies):.6f}")
    print(f"  Max: {max(entropies):.6f}")
    print(f"  Mean: {np.mean(entropies):.6f}")
    print(f"  Std: {np.std(entropies):.6f}")
    
    print("\n[PHASE 4] Saving Results")
    print("-" * 80)
    
    # Save results
    results = {
        'algorithm': 'QHIL (Quantum Human-in-the-Loop)',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_qubits': optimizer.n_qubits,
            'max_depth': optimizer.max_depth,
            'iterations': len(history),
        },
        'feedback_sequence': feedback_sequence,
        'history': [state.to_dict() for state in history],
        'statistics': {
            'fidelity': {
                'min': float(min(fidelities)),
                'max': float(max(fidelities)),
                'mean': float(np.mean(fidelities)),
                'std': float(np.std(fidelities)),
            },
            'entropy': {
                'min': float(min(entropies)),
                'max': float(max(entropies)),
                'mean': float(np.mean(entropies)),
                'std': float(np.std(entropies)),
            }
        }
    }
    
    with open('data/logs/qhil_demo_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✓ Results saved to: data/logs/qhil_demo_results.json")
    
    print("\n[PHASE 5] QHIL Communication Protocol")
    print("-" * 80)
    print("Quantum-Human Interaction Language (QHIL):")
    print("\nState Descriptors:")
    for name, symbol in QuantumHumanInteractionLanguage.DESCRIPTORS.items():
        print(f"  {symbol:15} → {name}")
    
    print("\nHuman Commands:")
    for name, cmd in QuantumHumanInteractionLanguage.COMMANDS.items():
        print(f"  {cmd:15} → {name}")
    
    print("\nIntensity Modifiers:")
    print(f"  STRONG          → 0.80 intensity")
    print(f"  MEDIUM          → 0.50 intensity")
    print(f"  WEAK            → 0.30 intensity")
    
    print("\n" + "="*80)
    print("✓ QHIL DEMO COMPLETE")
    print("="*80)
    
    return results

if __name__ == "__main__":
    results = demo_qhil()
    print("\nTo run interactive mode, execute:")
    print("  python3 quantum_human_loop.py")

