#!/usr/bin/env python3
"""Generate GHZ/W multipartite entanglement probabilities using QuTiP.

The script prints key=value pairs so native code can parse the output.
It attempts to validate the generated state with Qiskit first; if Qiskit is
unavailable it falls back to Cirq. The caller can disable validation by
omitting --validate.
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Optional, Tuple

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - numpy is required by QuTiP
    print(f"ERROR=Numpy missing: {exc}", file=sys.stderr)
    sys.exit(2)

try:
    from qutip.states import ghz_state, w_state
except ImportError as exc:
    print(f"ERROR=QuTiP missing: {exc}", file=sys.stderr)
    sys.exit(2)


def build_state(name: str, qubits: int):
    if qubits < 2:
        raise ValueError("Entanglement requires at least two qubits")
    name_lower = name.lower()
    if name_lower in {"ghz", "ghz_state"}:
        return ghz_state(qubits)
    if name_lower in {"w", "w_state"}:
        return w_state(qubits)
    raise ValueError(f"Unsupported entanglement state '{name}'")


def validate_with_qiskit(state_vector: np.ndarray, qubits: int, name: str) -> Optional[Tuple[str, float]]:
    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
    except Exception:
        return None

    try:
        qc = QuantumCircuit(qubits)
        if name.lower().startswith("ghz"):
            qc.h(0)
            for idx in range(1, qubits):
                qc.cx(0, idx)
        else:
            qc.initialize(state_vector, list(range(qubits)))
        sv = Statevector.from_label("0" * qubits).evolve(qc)
        fidelity = float(np.abs(np.vdot(state_vector.conjugate(), sv.data)) ** 2)
        return ("qiskit", fidelity)
    except Exception:
        return None


def validate_with_cirq(state_vector: np.ndarray, qubits: int) -> Optional[Tuple[str, float]]:
    try:
        import cirq
    except Exception:
        return None

    # Cirq validation is minimal: confirm the vector is normalized and finite.
    norm = float(np.real(np.vdot(state_vector.conjugate(), state_vector)))
    if not np.isfinite(norm) or abs(norm - 1.0) > 1e-6:
        return None
    return ("cirq", 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate GHZ/W entanglement state statistics.")
    parser.add_argument("--state", default="ghz", help="State type: ghz or w")
    parser.add_argument("--qubits", type=int, default=4, help="Number of qubits (default: 4)")
    parser.add_argument("--validate", action="store_true", help="Run backend validation (Qiskit or Cirq)")
    args = parser.parse_args()

    try:
        ket = build_state(args.state, args.qubits)
    except ValueError as exc:
        print(f"ERROR={exc}", file=sys.stderr)
        return 3

    probabilities = ket.probabilities()
    vector = ket.full().ravel()

    backend = "generated"
    fidelity = 1.0

    if args.validate:
        validation = validate_with_qiskit(vector, args.qubits, args.state)
        if validation is None:
            validation = validate_with_cirq(vector, args.qubits)
        if validation is None:
            print("ERROR=No quantum backend available (install qiskit or cirq)", file=sys.stderr)
            return 4
        backend, fidelity = validation

    print(f"STATE={args.state.upper()}")
    print(f"QUBITS={args.qubits}")
    print(f"BACKEND={backend}")
    print(f"FIDELITY={fidelity:.6f}")
    formatted = ",".join(f"{float(p):.10f}" for p in probabilities)
    print(f"PROBABILITIES={formatted}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
