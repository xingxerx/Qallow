#!/usr/bin/env python3
"""
Quantum Echoes Algorithm - OTOC-based Verifiable Quantum Advantage
Implements the algorithm from Google's Willow chip demo (Oct 2025)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import random_unitary, Operator

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class QuantumEchoesEngine:
    """Core quantum echoes algorithm implementation."""
    
    def __init__(self, n_qubits: int = 5, seed: int = 42):
        """Initialize the quantum echoes engine.
        
        Args:
            n_qubits: Number of qubits in the system
            seed: Random seed for reproducibility
        """
        self.n_qubits = n_qubits
        self.seed = seed
        self.backend = AerSimulator(method='statevector', seed_simulator=seed)
        np.random.seed(seed)
    
    def run_quantum_echoes(
        self,
        t_steps: int = 4,
        perturb_qubit: int = 0,
        shots: int = 1024,
    ) -> Dict[str, float]:
        """Execute quantum echoes protocol.

        Args:
            t_steps: Number of forward/backward evolution steps
            perturb_qubit: Which qubit to perturb with Pauli-X
            shots: Number of measurement shots

        Returns:
            Dictionary with OTOC fidelity and metadata
        """
        if t_steps < 1:
            raise ValueError("t_steps must be >= 1")
        if perturb_qubit >= self.n_qubits:
            raise ValueError(f"perturb_qubit {perturb_qubit} >= n_qubits {self.n_qubits}")

        # Step 1: Create circuit
        qc = QuantumCircuit(self.n_qubits)

        # Step 2: Forward evolution - Random unitaries
        U_forward = [random_unitary(2**self.n_qubits) for _ in range(t_steps)]
        for U in U_forward:
            qc.unitary(Operator(U), range(self.n_qubits), label='U_fwd')

        # Step 3: Perturb butterfly qubit
        qc.x(perturb_qubit)

        # Step 4: Backward evolution - Reverse unitaries (conjugate transpose)
        for U in reversed(U_forward):
            # Convert to numpy array, take conjugate transpose, then back to Operator
            U_array = np.asarray(U)
            U_dagger = np.conj(U_array.T)
            qc.unitary(Operator(U_dagger), range(self.n_qubits), label='U_bwd')

        # Step 5: Save state for overlap calculation
        qc.save_state()

        # Execute
        compiled = transpile(qc, self.backend)
        job = self.backend.run(compiled, shots=shots)
        result = job.result()
        final_state = result.get_statevector()

        # Calculate OTOC as fidelity overlap
        initial_state = np.zeros(2**self.n_qubits, dtype=complex)
        initial_state[0] = 1.0  # |00...0>
        otoc_fidelity = np.abs(np.dot(initial_state.conj(), final_state))**2

        return {
            'otoc_fidelity': float(otoc_fidelity),
            'n_qubits': self.n_qubits,
            't_steps': t_steps,
            'perturb_qubit': perturb_qubit,
            'shots': shots,
        }
    
    def compute_echo_decay(
        self,
        max_t_steps: int = 5,
        perturb_qubit: int = 0,
    ) -> Dict[int, float]:
        """Compute OTOC decay over multiple time steps.
        
        Args:
            max_t_steps: Maximum number of time steps to simulate
            perturb_qubit: Which qubit to perturb
            
        Returns:
            Dictionary mapping t_steps -> otoc_fidelity
        """
        decay_curve = {}
        for t in range(1, max_t_steps + 1):
            result = self.run_quantum_echoes(t_steps=t, perturb_qubit=perturb_qubit)
            decay_curve[t] = result['otoc_fidelity']
        return decay_curve


def run_quantum_echoes_demo(
    n_qubits: int = 5,
    t_steps: int = 4,
    shots: int = 2048,
    log_path: Optional[str] = None,
) -> Dict:
    """Run quantum echoes demo with optional telemetry logging.
    
    Args:
        n_qubits: Number of qubits
        t_steps: Evolution time steps
        shots: Measurement shots
        log_path: Optional path to save telemetry CSV
        
    Returns:
        Results dictionary
    """
    logger.info(f"Initializing Quantum-Coherence Pipeline (Phase 11)...")
    logger.info(f"Configuration: n_qubits={n_qubits}, t_steps={t_steps}, shots={shots}")
    
    engine = QuantumEchoesEngine(n_qubits=n_qubits)
    
    # Run quantum echoes
    logger.info("Executing quantum echoes protocol...")
    result = engine.run_quantum_echoes(t_steps=t_steps, shots=shots)
    otoc_echo = result['otoc_fidelity']
    
    logger.info(f"Quantum Echo Strength (OTOC): {otoc_echo:.4f}")
    
    # Telemetry logging
    if log_path:
        import csv
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['phase', 'otoc_fidelity', 'ticks', 'ethics_delta'])
            if f.tell() == 0:  # Write header if file is empty
                writer.writeheader()
            writer.writerow({
                'phase': 11,
                'otoc_fidelity': otoc_echo,
                'ticks': t_steps * shots,
                'ethics_delta': float(otoc_echo - 0.5),
            })
        logger.info(f"Telemetry logged to {log_path}")
    
    # Phase 14 handoff logic
    if otoc_echo >= 0.981:
        logger.info(f"[OK] Echo captured: Fidelity {otoc_echo:.3f} – Ready for Phase 14 coherence lattice.")
        result['phase14_ready'] = True
    else:
        logger.warning(f"[WARN] Decoherence detected: Fidelity {otoc_echo:.3f} – May need Phase 6 overlay control.")
        result['phase14_ready'] = False
    
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quantum Echoes Demo")
    parser.add_argument("--n-qubits", type=int, default=5, help="Number of qubits")
    parser.add_argument("--t-steps", type=int, default=4, help="Evolution time steps")
    parser.add_argument("--shots", type=int, default=2048, help="Measurement shots")
    parser.add_argument("--log", type=str, help="Telemetry log path")
    args = parser.parse_args()
    
    result = run_quantum_echoes_demo(
        n_qubits=args.n_qubits,
        t_steps=args.t_steps,
        shots=args.shots,
        log_path=args.log,
    )
    print(json.dumps(result, indent=2))

