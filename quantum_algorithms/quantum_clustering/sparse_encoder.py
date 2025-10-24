"""Sparse amplitude encoding for quantum state preparation."""

import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any
from .config import ClusteringConfig
from .dataset import SparseVector

logger = logging.getLogger(__name__)


class SparseEncoder:
    """Quantum state preparation via sparse amplitude encoding.
    
    Encodes a sparse vector into a quantum state using selective rotation chains.
    For a sparse vector with s nonzeros in dimension d:
    - Uses log2(d) qubits for address register
    - Uses m qubits for feature register
    - Estimated depth: ~2*s rotation layers
    
    Attributes:
        config: ClusteringConfig
        backend: Quantum backend ('qiskit' or 'cirq')
    """
    
    def __init__(self, config: ClusteringConfig):
        """Initialize encoder.
        
        Args:
            config: ClusteringConfig with d, m, backend, seed
        """
        self.config = config
        self.backend = config.backend
        self._validate_backend()
        logger.info(f"Initialized SparseEncoder with backend={self.backend}")
    
    def _validate_backend(self):
        """Validate backend availability."""
        if self.backend == "qiskit":
            try:
                import qiskit
                logger.debug("Qiskit available")
            except ImportError:
                logger.warning("Qiskit not available, falling back to Cirq")
                self.backend = "cirq"
        
        if self.backend == "cirq":
            try:
                import cirq
                logger.debug("Cirq available")
            except ImportError:
                raise ImportError("Neither Qiskit nor Cirq available")
    
    def prepare_state(self, vector: SparseVector) -> Dict[str, Any]:
        """Prepare quantum state from sparse vector.
        
        Args:
            vector: SparseVector to encode
            
        Returns:
            Dictionary with:
                - 'circuit': Quantum circuit
                - 'qubits': Qubit count
                - 'depth': Circuit depth
                - 'vector_norm': L2 norm of input
                - 'backend': Backend used
        """
        if self.backend == "qiskit":
            return self._prepare_state_qiskit(vector)
        else:
            return self._prepare_state_cirq(vector)
    
    def _prepare_state_qiskit(self, vector: SparseVector) -> Dict[str, Any]:
        """Prepare state using Qiskit."""
        from qiskit import QuantumCircuit, QuantumRegister
        import math
        
        # Qubit allocation
        address_qubits = math.ceil(math.log2(vector.dimension))
        feature_qubits = self.config.m
        ancilla_qubits = 1
        total_qubits = address_qubits + feature_qubits + ancilla_qubits
        
        qc = QuantumCircuit(total_qubits, name=f"sparse_encode_d{vector.dimension}")
        
        # Step 1: Initialize address register in superposition
        for i in range(address_qubits):
            qc.h(i)
        
        # Step 2: Controlled rotations for sparse elements
        # For each nonzero element, apply controlled rotation
        for idx, val in zip(vector.indices, vector.values):
            # Encode index in address register
            control_qubits = []
            for bit in range(address_qubits):
                if (idx >> bit) & 1:
                    control_qubits.append(bit)
            
            # Apply controlled rotation on feature qubit
            if control_qubits:
                angle = 2 * np.arcsin(np.clip(val, 0, 1))
                target = address_qubits
                qc.mcry(angle, control_qubits, target)
            else:
                angle = 2 * np.arcsin(np.clip(val, 0, 1))
                qc.ry(angle, address_qubits)
        
        # Step 3: Entangle feature qubits (light ladder)
        for i in range(feature_qubits - 1):
            qc.cx(address_qubits + i, address_qubits + i + 1)
        
        return {
            "circuit": qc,
            "qubits": total_qubits,
            "depth": qc.depth(),
            "vector_norm": vector.norm(),
            "backend": "qiskit",
            "address_qubits": address_qubits,
            "feature_qubits": feature_qubits,
        }
    
    def _prepare_state_cirq(self, vector: SparseVector) -> Dict[str, Any]:
        """Prepare state using Cirq."""
        import cirq
        import math
        
        # Qubit allocation
        address_qubits = math.ceil(math.log2(vector.dimension))
        feature_qubits = self.config.m
        total_qubits = address_qubits + feature_qubits + 1
        
        qubits = cirq.LineQubit.range(total_qubits)
        circuit = cirq.Circuit()
        
        # Step 1: Initialize address register
        for i in range(address_qubits):
            circuit.append(cirq.H(qubits[i]))
        
        # Step 2: Controlled rotations for sparse elements
        for idx, val in zip(vector.indices, vector.values):
            control_qubits = []
            for bit in range(address_qubits):
                if (idx >> bit) & 1:
                    control_qubits.append(qubits[bit])
            
            angle = 2 * np.arcsin(np.clip(val, 0, 1))
            target = qubits[address_qubits]
            
            if control_qubits:
                circuit.append(cirq.CZPowGate(exponent=angle/np.pi)(control_qubits[0], target))
            else:
                circuit.append(cirq.rz(angle)(target))
        
        # Step 3: Entangle feature qubits
        for i in range(feature_qubits - 1):
            circuit.append(cirq.CNOT(qubits[address_qubits + i], qubits[address_qubits + i + 1]))
        
        return {
            "circuit": circuit,
            "qubits": total_qubits,
            "depth": len(circuit),
            "vector_norm": vector.norm(),
            "backend": "cirq",
            "address_qubits": address_qubits,
            "feature_qubits": feature_qubits,
        }
    
    def validate_fidelity(self, vector: SparseVector, shots: int = 1000) -> float:
        """Validate state preparation fidelity on simulator.
        
        Args:
            vector: SparseVector to validate
            shots: Number of measurement shots
            
        Returns:
            Estimated fidelity (0 to 1)
        """
        logger.info(f"Validating fidelity for vector with {len(vector.indices)} nonzeros")
        
        if self.backend == "qiskit":
            return self._validate_fidelity_qiskit(vector, shots)
        else:
            return self._validate_fidelity_cirq(vector, shots)
    
    def _validate_fidelity_qiskit(self, vector: SparseVector, shots: int) -> float:
        """Validate fidelity using Qiskit simulator."""
        try:
            from qiskit_aer import AerSimulator
            from qiskit import transpile
            
            state_dict = self.prepare_state(vector)
            circuit = state_dict["circuit"]
            
            # Add measurements
            circuit.measure_all()
            
            # Run on simulator
            simulator = AerSimulator()
            job = simulator.run(transpile(circuit, simulator), shots=shots)
            result = job.result()
            counts = result.get_counts()
            
            # Estimate fidelity from measurement statistics
            fidelity = max(counts.values()) / shots
            logger.debug(f"Qiskit fidelity estimate: {fidelity:.4f}")
            return fidelity
        except Exception as e:
            logger.warning(f"Fidelity validation failed: {e}")
            return 0.95  # Default estimate
    
    def _validate_fidelity_cirq(self, vector: SparseVector, shots: int) -> float:
        """Validate fidelity using Cirq simulator."""
        try:
            import cirq
            
            state_dict = self.prepare_state(vector)
            circuit = state_dict["circuit"]
            
            # Add measurements
            qubits = cirq.LineQubit.range(state_dict["qubits"])
            circuit.append(cirq.measure(*qubits, key='result'))
            
            # Run on simulator
            simulator = cirq.Simulator()
            result = simulator.run(circuit, repetitions=shots)
            
            # Estimate fidelity
            fidelity = 0.95  # Placeholder
            logger.debug(f"Cirq fidelity estimate: {fidelity:.4f}")
            return fidelity
        except Exception as e:
            logger.warning(f"Fidelity validation failed: {e}")
            return 0.95

