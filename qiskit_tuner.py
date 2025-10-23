# qiskit_tuner.py
from qiskit import QuantumCircuit
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
import numpy as np, json, sys

N = int(sys.argv[1]) if len(sys.argv)>1 else 8
p = int(sys.argv[2]) if len(sys.argv)>2 else 2

# Simple ring Ising model; you can replace with Phase12/13-derived weights
J = np.ones((N,N))*0.0
for i in range(N):
    J[i,(i+1)%N] = 1.0

# Cost Hamiltonian H = -sum_{i<j} J_ij Z_i Z_j
paulis, coeffs = [], []
for i in range(N):
    for j in range(i+1, N):
        if abs(J[i,j])>1e-12:
            z = ['I']*N
            z[i]=z[j]='Z'
            paulis.append(''.join(z)[::-1])
            coeffs.append(-J[i,j])
H = SparsePauliOp.from_list(list(zip(paulis, coeffs)))

# QAOA ansatz
def qaoa_circuit(gammas, betas):
    qc = QuantumCircuit(N)
    qc.h(range(N))
    for layer in range(len(gammas)):
        gamma, beta = gammas[layer], betas[layer]
        # cost mixer (ZZ rotations)
        for i in range(N):
            j=(i+1)%N
            qc.cx(i,j); qc.rz(2*gamma, j); qc.cx(i,j)
        # X mixer
        for q in range(N):
            qc.rx(2*beta, q)
    return qc

# crude coordinate descent
L = p
gammas = np.full(L, 0.5)
betas  = np.full(L, 0.5)
est = Estimator()

def energy(gammas, betas):
    qc = qaoa_circuit(gammas, betas)
    return est.run(qc, H).result().values[0]

E = energy(gammas, betas)
for it in range(60):
    for k in range(L):
        for step in (+0.1, -0.1):
            trial = gammas.copy(); trial[k] += step
            En = energy(trial, betas)
            if En < E: E, gammas = En, trial
        for step in (+0.1, -0.1):
            trial = betas.copy(); trial[k] += step
            En = energy(gammas, trial)
            if En < E: E, betas = En, trial

# Derive per-node effective gain from expectation of ZZ links
# Map expected alignment -> gain in [0.001, 0.01]
gain_base = 0.001
gain_span = 0.009
align = min(1.0, max(0.0, -E/(N)))  # rough normalization
alpha_eff = gain_base + gain_span*align

out = {"alpha_eff": float(alpha_eff), "E": float(E), "N": N, "p": p}
print(json.dumps(out))