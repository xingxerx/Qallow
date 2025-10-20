#!/usr/bin/env python3
"""Qallow ↔ Qiskit bridge utility.

Transforms ternary topology states from the Phase 11 coherence pipeline into a
small Qiskit circuit, executes it on the configured backend, and prints a single
coherence metric in the form `coherence=<value>` to stdout. The script prefers
IBM Quantum backends when credentials are available and falls back to the local
Aer simulator otherwise.

Environment variables:
  * QISKIT_IBM_TOKEN      – Optional. Token used to authenticate with IBM
                            Quantum (if not already stored locally).
  * QALLOW_QISKIT_BACKEND – Optional. Explicit backend name to target.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict, Iterable, List, Optional

try:
    from qiskit import QuantumCircuit, transpile
except ImportError as exc:  # pragma: no cover - dependency missing in CI
    sys.stderr.write("[qiskit_bridge] Qiskit is not installed: %s\n" % exc)
    raise

try:  # Prefer IBM Runtime when available
    from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
    HAVE_IBM_RUNTIME = True
except ImportError:  # pragma: no cover - optional dependency
    QiskitRuntimeService = None  # type: ignore
    Sampler = None  # type: ignore
    HAVE_IBM_RUNTIME = False

try:
    from qiskit_aer import AerSimulator
except ImportError:  # pragma: no cover - optional dependency
    AerSimulator = None  # type: ignore


TOPOLOGY_TOKENS = {"1", "0", "-1", "N"}


def parse_states(raw: str) -> List[str]:
    tokens = [token.strip() for token in raw.split(',') if token.strip()]
    for token in tokens:
        if token not in TOPOLOGY_TOKENS:
            raise ValueError(f"Unsupported topology token '{token}'. Expected one of {sorted(TOPOLOGY_TOKENS)}")
    if not tokens:
        raise ValueError("At least one topology token is required")
    return tokens


def build_circuit(tokens: Iterable[str]) -> QuantumCircuit:
    tokens_list = list(tokens)
    qc = QuantumCircuit(len(tokens_list))
    for idx, token in enumerate(tokens_list):
        if token == "1":
            qc.x(idx)
        elif token == "-1":
            qc.h(idx)
            qc.z(idx)
        elif token == "N":
            qc.h(idx)
        else:  # "0" keeps |0>
            pass
    qc.measure_all()
    return qc


def resolve_backend_label(raw) -> str:
    if callable(raw):  # pragma: no cover - defensive branch
        try:
            resolved = raw()
        except TypeError:
            resolved = raw
    else:
        resolved = raw
    return str(resolved)


def select_backend(args: argparse.Namespace) -> Dict[str, object]:
    explicit_backend = args.backend or os.getenv("QALLOW_QISKIT_BACKEND")

    if HAVE_IBM_RUNTIME:
        try:
            service = QiskitRuntimeService()
            if explicit_backend:
                backend = service.backend(explicit_backend)
            else:
                backend = service.least_busy(simulator=args.allow_simulator)
            sampler = Sampler(backend=backend, options={"shots": args.shots})
            return {
                "mode": "ibm",
                "sampler": sampler,
                "backend_label": resolve_backend_label(getattr(backend, "name", backend)),
            }
        except Exception as exc:  # pragma: no cover - runtime failures
            sys.stderr.write(f"[qiskit_bridge] IBM Runtime unavailable, falling back to Aer: {exc}\n")

    if AerSimulator is None:
        raise RuntimeError("qiskit-aer is not available and IBM Runtime could not be used")

    backend = AerSimulator()
    return {
        "mode": "aer",
        "backend": backend,
        "backend_label": resolve_backend_label(getattr(backend, "name", "AerSimulator")),
    }


def execute_circuit(qc: QuantumCircuit, selection: Dict[str, object], shots: int) -> Dict[str, float]:
    mode = selection.get("mode")
    if mode == "ibm":
    sampler = selection["sampler"]  # type: ignore[index]
        job = sampler.run([qc])
        quasi = job.result().quasi_dists[0]
        binary = quasi.binary_probabilities()
        return {state: float(prob) for state, prob in binary.items()}

    if mode == "aer":
        backend = selection["backend"]  # type: ignore[index]
        transpiled = transpile(qc, backend)
        job = backend.run(transpiled, shots=shots)
        counts = job.result().get_counts()
        total = float(sum(counts.values())) or 1.0
        return {state: count / total for state, count in counts.items()}

    raise RuntimeError(f"Unsupported backend mode '{mode}' for Qiskit execution")


def compute_coherence(probabilities: Dict[str, float], qubits: int) -> float:
    if not probabilities:
        return 0.0
    aligned = probabilities.get("0" * qubits, 0.0) + probabilities.get("1" * qubits, 0.0)
    entropy = -sum(p * math.log(p, 2) for p in probabilities.values() if p > 0)
    max_entropy = qubits
    entropy_score = 1.0 - entropy / max_entropy if max_entropy else 0.0
    coherence = 0.6 * aligned + 0.4 * entropy_score
    return max(0.0, min(1.0, coherence))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Execute Qallow topology states on a Qiskit backend")
    parser.add_argument("--states", required=True, help="Comma-separated topology tokens (-1,0,1,N)")
    parser.add_argument("--shots", type=int, default=512, help="Number of shots to execute (default: 512)")
    parser.add_argument("--backend", help="Explicit backend name to target")
    parser.add_argument("--allow-simulator", action="store_true", help="Allow selection of simulator backends")
    args = parser.parse_args(argv)

    tokens = parse_states(args.states)
    qc = build_circuit(tokens)
    selection = select_backend(args)
    backend_label = resolve_backend_label(selection.get("backend_label", "unknown"))
    sys.stderr.write(f"[qiskit_bridge] backend={backend_label}\n")
    sys.stderr.flush()
    probabilities = execute_circuit(qc, selection, args.shots)
    coherence = compute_coherence(probabilities, len(tokens))

    sys.stdout.write(f"coherence={coherence:.6f}\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # pragma: no cover - top-level failure path
        sys.stderr.write(f"[qiskit_bridge] Execution failed: {exc}\n")
        sys.exit(1)
