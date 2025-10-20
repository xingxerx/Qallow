#!/usr/bin/env python3
"""
Execute a Bell-state circuit using IBM Quantum hardware when available, with Aer fallback.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit_ibm_runtime.exceptions import IBMRuntimeError

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

try:
    from qiskit_aer import AerSimulator
except ImportError:  # pragma: no cover - optional dependency
    AerSimulator = None


def load_api_token() -> Optional[str]:
    """Load the IBM Quantum API token from the environment."""
    if load_dotenv is not None:
        load_dotenv()
    return os.getenv("QISKIT_IBM_TOKEN")


def build_bell_circuit() -> QuantumCircuit:
    """Construct a Bell-state circuit."""
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])
    return circuit


def run_on_hardware(
    circuit: QuantumCircuit,
    shots: int,
    token: Optional[str],
) -> tuple[dict[str, float], str]:
    """Attempt to run the circuit on IBM Quantum hardware."""
    try:
        if token:
            service = QiskitRuntimeService(channel="ibm_quantum", token=token)
        else:
            service = QiskitRuntimeService()

        backend = service.least_busy(operational=True, simulator=False)
        sampler = Sampler(backend=backend)
        job = sampler.run([circuit], shots=shots)
        quasi_dist = job.result().quasi_dists[0].binary_probabilities()
        return {state: float(prob) for state, prob in quasi_dist.items()}, backend.name
    except Exception as error:  # pragma: no cover - runtime fallback
        raise RuntimeError("Unable to execute on IBM Quantum hardware.") from error


def run_on_aer(circuit: QuantumCircuit, shots: int) -> tuple[dict[str, float], str]:
    """Run the circuit locally using Qiskit Aer."""
    if AerSimulator is None:
        raise RuntimeError(
            "IBM Quantum authentication failed and qiskit-aer is not installed for fallback."
        )

    backend = AerSimulator()
    result = backend.run(circuit, shots=shots).result()
    counts = result.get_counts()
    return {state: float(count) / float(shots) for state, count in counts.items()}, "aer_simulator"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=1024, help="Number of shots to execute.")
    parser.add_argument(
        "--hardware-only",
        action="store_true",
        help="Fail instead of falling back to Aer when hardware execution is unavailable.",
    )
    args = parser.parse_args()

    circuit = build_bell_circuit()
    token = load_api_token()

    try:
        counts, backend_name = run_on_hardware(circuit, args.shots, token)
        source = "hardware"
    except (IBMRuntimeError, RuntimeError) as error:
        if args.hardware_only:
            raise
        counts, backend_name = run_on_aer(circuit, args.shots)
        source = "aer"
        print(f"Hardware execution unavailable ({error}); using Aer simulator instead.")

    print(f"Backend: {backend_name} (source={source})")
    for state, probability in sorted(counts.items()):
        print(f"{state}: {probability:.4f}")


if __name__ == "__main__":
    main()
