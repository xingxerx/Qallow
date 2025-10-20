#!/usr/bin/env python3
"""
Example: run a Bell state circuit on IBM Quantum hardware via Qiskit Runtime.

Configure the IBM Quantum API token in your environment (e.g., .env) under
`QISKIT_IBM_TOKEN` before executing this script.
"""

from __future__ import annotations

import os
from typing import Optional

from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None


def load_api_token() -> Optional[str]:
    """Load the IBM Quantum API token from environment variables."""
    if load_dotenv is not None:
        load_dotenv()
    return os.getenv("QISKIT_IBM_TOKEN")


def build_bell_circuit() -> QuantumCircuit:
    """Return a Bell state circuit."""
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])
    return circuit


def main() -> None:
    token = load_api_token()
    if token:
        service = QiskitRuntimeService(channel="ibm_quantum", token=token)
    else:
        # Fall back to locally saved credentials
        service = QiskitRuntimeService()

    backend = service.least_busy(operational=True, simulator=False)
    print(f"Using backend: {backend.name}")

    circuit = build_bell_circuit()
    sampler = Sampler(backend=backend)
    job = sampler.run([circuit])
    result = job.result()
    quasi_dist = result.quasi_dists[0]
    print("Quasi-probability distribution:", quasi_dist)


if __name__ == "__main__":
    main()
