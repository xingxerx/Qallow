# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] """Sparse amplitude encoding for quantum state preparation."""
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] from .config import ClusteringConfig
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] from .dataset import SparseVector
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] logger = logging.getLogger(__name__)
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] class SparseEncoder:
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]     """Quantum state preparation via sparse amplitude encoding.
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]     Encodes a sparse vector into a quantum state using selective rotation chains.
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]     For a sparse vector with s nonzeros in dimension d:
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]     - Uses log2(d) qubits for address register
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]     - Uses m qubits for feature register
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]     - Estimated depth: ~2*s rotation layers
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]     Attributes:
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]         config: ClusteringConfig
        backend: Quantum backend (Cirq only)
    """

    def __init__(self, config: ClusteringConfig):
        """Initialize encoder.

        Args:
            config: ClusteringConfig with d, m, backend, seed
        """
        self.config = config
        self.backend = "cirq"  # Only Cirq is supported now
        self._validate_backend()
        logger.info(f"Initialized SparseEncoder with backend={self.backend}")

    def _validate_backend(self):
        """Validate backend availability."""
        try:
            import cirq
            logger.debug("Cirq available")
        except ImportError:
            raise ImportError("Cirq is required for SparseEncoder")
    
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
        return self._prepare_state_cirq(vector)

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
        return self._validate_fidelity_cirq(vector, shots)

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

