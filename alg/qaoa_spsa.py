#!/usr/bin/env python3
"""
QAOA Optimizer for Qallow using Cirq and SPSA.

This script finds optimal parameters for a QAOA circuit that minimizes the
energy of an Ising Hamiltonian. The results are used to determine control
gains for the main Qallow application.
"""

import argparse
import json
from datetime import datetime
import numpy as np
import cirq

def load_ising_model(config_path):
    """Load Ising model from a JSON configuration file."""
    with open(config_path, "r") as f:
        config = json.load(f)

    N = config.get("N", 8)
    J = np.zeros((N, N))

    # Load coupling matrix from CSV if provided
    csv_j = config.get("csv_j")
    if csv_j:
        try:
            with open(csv_j, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(",")
                    i, j, coupling = int(parts[0]), int(parts[1]), float(parts[2])
                    J[i, j] = coupling
                    J[j, i] = coupling
        except Exception as e:
            print(f"[QAOA-Cirq] Warning: Could not load CSV '{csv_j}': {e}. Falling back to default.")
            # Fallback to default ring topology
            for i in range(N):
                j = (i + 1) % N
                J[i, j] = 1.0
                J[j, i] = 1.0
    else:
        # Default: ring topology
        for i in range(N):
            j = (i + 1) % N
            J[i, j] = 1.0
            J[j, i] = 1.0

    return N, J, config

def qaoa_circuit(gamma, beta, J, N):
    """Constructs the QAOA circuit in Cirq."""
    p = len(gamma)
    qubits = cirq.LineQubit.range(N)
    circuit = cirq.Circuit()

    # Initial state: superposition
    circuit.append(cirq.H.on_each(*qubits))

    # Apply p layers of cost and mixer unitaries
    for i in range(p):
        # Cost unitary
        for j in range(N):
            for k in range(j + 1, N):
                if J[j, k] != 0:
                    circuit.append(cirq.ZZ(qubits[j], qubits[k])**(2 * gamma[i] * J[j, k] / np.pi))
        
        # Mixer unitary
        circuit.append(cirq.X.on_each(*qubits)**(beta[i] / np.pi))

    return circuit

def ising_energy_from_counts(counts, J):
    """Compute Ising energy from measurement counts."""
    total_shots = sum(counts.values())
    avg_energy = 0.0
    
    if not counts:
        return 0.0

    N = len(next(iter(counts.keys())))

    for bitstring, count in counts.items():
        z = np.array([1 if bit == '0' else -1 for bit in bitstring])
        energy = 0
        for i in range(N):
            for j in range(i + 1, N):
                energy -= J[i, j] * z[i] * z[j]
        avg_energy += energy * count

    return avg_energy / total_shots if total_shots > 0 else 0.0

def get_qaoa_energy(params, J, N, shots=1000):
    """Execute the QAOA circuit and compute the energy."""
    p = len(params) // 2
    gamma = params[:p]
    beta = params[p:]

    circuit = qaoa_circuit(gamma, beta, J, N)
    circuit.append(cirq.measure(*cirq.LineQubit.range(N), key='result'))

    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=shots)
    
    # Convert measurements to counts dictionary
    measurements = result.measurements['result']
    bitstrings = [''.join(map(str, row)) for row in measurements]
    counts = {k: v for k, v in zip(*np.unique(bitstrings, return_counts=True))}

    return ising_energy_from_counts(counts, J)

def spsa_optimizer(J, N, config):
    """SPSA optimizer to find optimal QAOA parameters."""
    p = config.get("p", 2)
    iterations = config.get("spsa_iterations", 50)
    a = config.get("spsa_a", 0.1)
    c = config.get("spsa_c", 0.1)
    
    params = np.random.rand(2 * p) * np.pi
    best_energy = float('inf')
    best_params = params.copy()

    print(f"[QAOA-Cirq] Starting SPSA optimization ({iterations} iterations)...")

    for iteration in range(iterations):
        delta = np.random.choice([-1, 1], size=len(params))
        
        params_plus = params + c * delta
        params_minus = params - c * delta
        
        energy_plus = get_qaoa_energy(params_plus, J, N)
        energy_minus = get_qaoa_energy(params_minus, J, N)
        
        gradient = (energy_plus - energy_minus) / (2 * c * delta)
        
        step_size = a / (iteration + 1) ** 0.602
        params -= step_size * gradient
        
        current_energy = (energy_plus + energy_minus) / 2
        if current_energy < best_energy:
            best_energy = current_energy
            best_params = params.copy()
        
        if (iteration + 1) % 10 == 0:
            print(f"[QAOA-Cirq] Iteration {iteration + 1:3d}: Energy = {best_energy:.6f}")
    
    return best_energy, best_params

def map_energy_to_gain(energy, J, alpha_min, alpha_max):
    """Map Ising energy to control gain."""
    # Normalize energy based on a rough estimate of the energy range
    max_possible_energy = np.sum(np.abs(J))
    if max_possible_energy == 0:
        return alpha_min

    normalized = max(0, min(1, (energy + max_possible_energy) / (2 * max_possible_energy)))
    
    alpha_eff = alpha_max - normalized * (alpha_max - alpha_min)
    return alpha_eff

def run_qaoa_optimizer(config_path):
    """Main QAOA optimizer function."""
    print(f"\n[QAOA-Cirq] Loading config: {config_path}")
    N, J, config = load_ising_model(config_path)
    
    print(f"[QAOA-Cirq] System size: N={N}")
    print(f"[QAOA-Cirq] QAOA depth: p={config.get('p', 2)}")
    
    best_energy, best_params = spsa_optimizer(J, N, config)
    
    alpha_min = config.get("alpha_min", 0.001)
    alpha_max = config.get("alpha_max", 0.01)
    alpha_eff = map_energy_to_gain(best_energy, J, alpha_min, alpha_max)
    
    result = {
        "energy": float(best_energy),
        "alpha_eff": float(alpha_eff),
        "iterations": config.get("spsa_iterations", 50),
        "system_size": N,
        "qaoa_depth": config.get('p', 2),
        "timestamp": datetime.now().isoformat(),
        "config_path": config_path,
        "backend": "cirq"
    }
    
    print(f"\n[QAOA-Cirq] Optimization complete")
    print(f"[QAOA-Cirq] Best energy: {best_energy:.6f}")
    print(f"[QAOA-Cirq] Mapped gain: {alpha_eff:.6f}")
    
    return result

def main():
    parser = argparse.ArgumentParser(description="QAOA Optimizer with Cirq")
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the JSON config file for the Ising model."
    )
    args = parser.parse_args()
    
    result = run_qaoa_optimizer(args.config_path)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()

