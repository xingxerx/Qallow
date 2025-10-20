"""Utilities to bridge Qallow ternary state experiments with IBM Quantum services."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, MutableMapping, Optional, Sequence

from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit_ibm_runtime.exceptions import IBMRuntimeError

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

try:
    from qiskit_aer import AerSimulator
except ImportError:  # pragma: no cover - qiskit-aer is optional but recommended
    AerSimulator = None


@dataclass(slots=True)
class TernaryResult:
    """Container for quantum execution results consumed by Qallow telemetry."""

    counts: Mapping[str, float]
    backend_name: str
    shots: int
    source: str  # "hardware" or "aer"


def _load_token_from_env() -> Optional[str]:
    """Return the IBM Quantum token from environment variables if available."""
    if load_dotenv is not None:
        load_dotenv()  # load .env files when present
    return os.getenv("QISKIT_IBM_TOKEN")


def _get_runtime_service(explicit_token: Optional[str] = None) -> QiskitRuntimeService:
    """
    Initialize QiskitRuntimeService using an explicit token, an env token, or saved credentials.

    Preference order:
    1. explicit_token argument
    2. QISKIT_IBM_TOKEN environment variable (supports .env files)
    3. Locally saved credentials via QiskitRuntimeService.save_account
    """
    try:
        if explicit_token:
            return QiskitRuntimeService(channel="ibm_quantum", token=explicit_token)

        env_token = _load_token_from_env()
        if env_token:
            return QiskitRuntimeService(channel="ibm_quantum", token=env_token)

        # Fall back to default credential lookup (disk store)
        return QiskitRuntimeService()
    except Exception as error:  # pragma: no cover - defensive guard
        raise RuntimeError("Unable to initialize QiskitRuntimeService.") from error


def build_ternary_circuit(ternary_states: Sequence[int]) -> QuantumCircuit:
    """
    Build a simple circuit representing ternary (-1, 0, 1) states on qubits.

    The mapping is heuristic: 0 leaves the qubit idle, 1 applies an X gate, and -1
    applies an H-Z-H sequence to approximate a negative phase. This placeholder
    can be replaced by a true qutrit encoding when Phase 11 formalizes its mapping.
    """
    num_qubits = max(len(ternary_states), 1)
    circuit = QuantumCircuit(num_qubits, num_qubits)

    for index, state in enumerate(ternary_states):
        qubit = index % num_qubits
        if state > 0:
            circuit.x(qubit)
        elif state < 0:
            circuit.h(qubit)
            circuit.z(qubit)
            circuit.h(qubit)
        # state == 0 leaves the qubit untouched

    circuit.barrier()
    circuit.measure(range(num_qubits), range(num_qubits))
    return circuit


def _execute_on_hardware(
    circuit: QuantumCircuit,
    service: QiskitRuntimeService,
    shots: int,
) -> TernaryResult:
    backend = service.least_busy(operational=True, simulator=False)
    sampler = Sampler(backend=backend)
    job = sampler.run([circuit], shots=shots)
    result = job.result()
    quasi_dist = result.quasi_dists[0].binary_probabilities()
    counts: MutableMapping[str, float] = {state: float(prob) for state, prob in quasi_dist.items()}

    return TernaryResult(
        counts=dict(counts),
        backend_name=backend.name,
        shots=shots,
        source="hardware",
    )


def _execute_on_aer(circuit: QuantumCircuit, shots: int) -> TernaryResult:
    if AerSimulator is None:
        raise RuntimeError(
            "qiskit-aer is not installed; install it or provide access to IBM Quantum hardware."
        )

    backend = AerSimulator()
    job = backend.run(circuit, shots=shots)
    result = job.result()
    raw_counts = result.get_counts()
    counts: MutableMapping[str, float] = {
        bitstring: float(value) / float(shots) for bitstring, value in raw_counts.items()
    }

    return TernaryResult(
        counts=dict(counts),
        backend_name="aer_simulator",
        shots=shots,
        source="aer",
    )


def run_ternary_sim(
    ternary_states: Sequence[int],
    *,
    shots: int = 1024,
    prefer_hardware: bool = True,
    token: Optional[str] = None,
) -> TernaryResult:
    """
    Execute a ternary-to-qubit circuit using IBM Quantum hardware with Aer fallback.

    Args:
        ternary_states: Iterable containing ternary (-1, 0, 1) values derived from Qallow.
        shots: Number of measurements to request.
        prefer_hardware: When True, attempt to route jobs to real hardware first.
        token: Optional IBM Quantum token override (otherwise use env or saved credentials).

    Returns:
        TernaryResult with normalized counts, backend information, and execution source.
    """
    circuit = build_ternary_circuit(ternary_states)

    if prefer_hardware:
        try:
            service = _get_runtime_service(explicit_token=token)
            return _execute_on_hardware(circuit, service, shots)
        except (IBMRuntimeError, RuntimeError) as error:
            if AerSimulator is None:
                raise RuntimeError(
                    "IBM Quantum hardware execution failed and qiskit-aer is unavailable."
                ) from error
        except Exception as _error:  # pragma: no cover - defensive fallback
            if AerSimulator is None:
                raise
            # Unexpected issue (e.g., temporary network interruption); fall back to Aer.
            # Downstream telemetry can note the source to distinguish from hardware runs.
            pass

    return _execute_on_aer(circuit, shots)


__all__ = [
    "TernaryResult",
    "build_ternary_circuit",
    "run_ternary_sim",
]
