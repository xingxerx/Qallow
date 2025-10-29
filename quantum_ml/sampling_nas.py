# qallow/quantum_ml/sampling_nas.py
import numpy as np
from qallow.phase11 import quantum_coherence_pipeline
from qallow.algorithms import QAOASampler

class QuantumNASExplorer:
    def __init__(self, search_space_dim=8):
        self.sampler = QAOASampler(qubits=search_space_dim, p=2)
        
    def generate_architectures(self, n_samples=100):
        # Quantum sampling in superposition of all architectures
        quantum_states = self.sampler.sample_superposition(shots=n_samples)
        
        # Map to neural architectures
        return [self._decode_architecture(state) for state in quantum_states]
        
    def _decode_architecture(self, quantum_state):
        # Bit string → layer config (conv/pool/dense)
        return {
            'layers': [(state[i:i+2], state[i+2:i+4]) for i in range(0, len(state), 4)],
            'skip_connections': state[-3:],
            'activation': 'gelu' if state[0] else 'relu'
        }
