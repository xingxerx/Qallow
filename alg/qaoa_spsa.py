"""
QAOA + SPSA Quantum Optimizer
Minimizes Ising Hamiltonian to find optimal control gain for Qallow
"""

import json
import numpy as np
from datetime import datetime


def load_ising_model(config_path):
    """Load Ising model from config"""
    with open(config_path, "r") as f:
        config = json.load(f)
    
    N = config.get("N", 8)
    csv_j = config.get("csv_j")
    
    # Initialize coupling matrix
    J = np.zeros((N, N))
    
    if csv_j:
        # Load from CSV
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
            print(f"[QAOA] Warning: Could not load CSV: {e}")
            # Fall back to default ring
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


def ising_energy(z, J):
    """Compute Ising Hamiltonian energy for spin configuration z"""
    energy = 0.0
    N = len(z)
    for i in range(N):
        for j in range(i + 1, N):
            energy -= J[i, j] * z[i] * z[j]
    return energy


def qaoa_circuit_energy(gamma, beta, J, N, shots=1000):
    """
    Simulate QAOA circuit and measure energy
    gamma, beta: QAOA parameters
    J: coupling matrix
    N: number of qubits
    shots: number of measurement shots
    """
    try:
        from qiskit import QuantumCircuit, QuantumRegister
        from qiskit_aer import AerSimulator
        
        qr = QuantumRegister(N, "q")
        qc = QuantumCircuit(qr)
        
        # Initial superposition
        for i in range(N):
            qc.h(qr[i])
        
        # Cost layer: ZZ interactions
        for i in range(N):
            for j in range(i + 1, N):
                if J[i, j] != 0:
                    qc.rzz(2 * gamma * J[i, j], qr[i], qr[j])
        
        # Mixer layer: X rotations
        for i in range(N):
            qc.rx(2 * beta, qr[i])
        
        # Measurement
        qc.measure_all()
        
        # Simulate
        simulator = AerSimulator()
        job = simulator.run(qc, shots=shots)
        result = job.result()
        counts = result.get_counts()
        
        # Compute average energy
        avg_energy = 0.0
        for bitstring, count in counts.items():
            z = np.array([1 if bit == "1" else -1 for bit in reversed(bitstring)])
            energy = ising_energy(z, J)
            avg_energy += energy * count / shots
        
        return avg_energy
        
    except ImportError:
        # Fallback: classical simulation
        print("[QAOA] Qiskit not available, using classical simulation")
        
        # Random sampling
        avg_energy = 0.0
        for _ in range(100):
            z = np.random.choice([-1, 1], size=N)
            energy = ising_energy(z, J)
            avg_energy += energy / 100
        
        return avg_energy


def spsa_optimizer(J, N, config):
    """
    SPSA (Simultaneous Perturbation Stochastic Approximation) optimizer
    Finds optimal QAOA parameters
    """
    p = config.get("p", 2)
    iterations = config.get("spsa_iterations", 50)
    a = config.get("spsa_a", 0.1)
    c = config.get("spsa_c", 0.1)
    
    # Initialize parameters
    params = np.random.rand(2 * p) * np.pi
    
    best_energy = float('inf')
    best_params = params.copy()
    
    print(f"[QAOA] Starting SPSA optimization ({iterations} iterations)...")
    
    for iteration in range(iterations):
        # Random perturbation
        delta = np.random.choice([-1, 1], size=len(params))
        
        # Evaluate at perturbed points
        params_plus = params + c * delta
        params_minus = params - c * delta
        
        # Compute energies
        gamma_plus = params_plus[:p]
        beta_plus = params_plus[p:]
        energy_plus = qaoa_circuit_energy(gamma_plus, beta_plus, J, N)
        
        gamma_minus = params_minus[:p]
        beta_minus = params_minus[p:]
        energy_minus = qaoa_circuit_energy(gamma_minus, beta_minus, J, N)
        
        # Gradient estimate
        gradient = (energy_plus - energy_minus) / (2 * c * delta)
        
        # Parameter update
        step_size = a / (iteration + 1) ** 0.602
        params -= step_size * gradient
        
        # Track best
        current_energy = (energy_plus + energy_minus) / 2
        if current_energy < best_energy:
            best_energy = current_energy
            best_params = params.copy()
        
        if (iteration + 1) % 10 == 0:
            print(f"[QAOA] Iteration {iteration + 1:3d}: Energy = {best_energy:.6f}")
    
    return best_energy, best_params


def map_energy_to_gain(energy, alpha_min, alpha_max):
    """
    Map Ising energy to control gain
    Lower energy -> higher gain (more aggressive tuning)
    """
    # Normalize energy to [0, 1]
    # Assume energy range is roughly [-N^2, 0]
    normalized = max(0, min(1, -energy / 100))
    
    # Map to gain range
    alpha_eff = alpha_min + normalized * (alpha_max - alpha_min)
    
    return alpha_eff


def run_qaoa_optimizer(config_path):
    """
    Main QAOA optimizer function
    Returns dict with energy, alpha_eff, and metadata
    """
    print(f"\n[QAOA] Loading config: {config_path}")
    
    # Load Ising model
    N, J, config = load_ising_model(config_path)
    
    print(f"[QAOA] System size: N={N}")
    print(f"[QAOA] QAOA depth: p={config.get('p', 2)}")
    
    # Run SPSA optimizer
    best_energy, best_params = spsa_optimizer(J, N, config)
    
    # Map energy to gain
    alpha_min = config.get("alpha_min", 0.001)
    alpha_max = config.get("alpha_max", 0.01)
    alpha_eff = map_energy_to_gain(best_energy, alpha_min, alpha_max)
    
    # Prepare result
    result = {
        "energy": float(best_energy),
        "alpha_eff": float(alpha_eff),
        "iterations": config.get("spsa_iterations", 50),
        "system_size": N,
        "qaoa_depth": config.get("p", 2),
        "timestamp": datetime.now().isoformat(),
        "config_path": config_path
    }
    
    print(f"\n[QAOA] Optimization complete")
    print(f"[QAOA] Best energy: {best_energy:.6f}")
    print(f"[QAOA] Mapped gain: {alpha_eff:.6f}")
    
    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        result = run_qaoa_optimizer(config_path)
        print(json.dumps(result, indent=2))

