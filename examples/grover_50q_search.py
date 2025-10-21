#!/usr/bin/env python3
"""
Example: 50-qubit Grover search with Aer MPS simulation.

Marks both |0...0⟩ and |1...1⟩, optimizes the circuit with a preset pass
manager, and evaluates the distribution with a runtime-compatible sampler.
"""

from __future__ import annotations

import argparse
import math
from typing import Iterable, List

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import GroverOperator, MCMT, ZGate
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

try:
    from qiskit_aer import AerSimulator
except ImportError as exc:  # pragma: no cover
    raise SystemExit("qiskit-aer is required for this example.") from exc

try:
    from qiskit_ibm_runtime import SamplerV2 as RuntimeSampler
except ImportError:  # pragma: no cover
    RuntimeSampler = None

try:
    from qiskit.primitives import BackendSampler
except ImportError as exc:  # pragma: no cover
    raise SystemExit("qiskit>=0.45 is required for BackendSampler support.") from exc


def grover_oracle(marked_states: Iterable[str], num_qubits: int) -> QuantumCircuit:
    """Return an oracle circuit that flips the phase of each marked basis state."""
    oracle = QuantumCircuit(num_qubits)
    mcmt = MCMT(ZGate(), num_qubits - 1, 1)

    for target in marked_states:
        if len(target) != num_qubits:
            raise ValueError("Marked state length must match num_qubits.")
        reversed_bits = target[::-1]
        zero_indices = [idx for idx, bit in enumerate(reversed_bits) if bit == "0"]

        if zero_indices:
            oracle.x(zero_indices)

        oracle.compose(mcmt, inplace=True)

        if zero_indices:
            oracle.x(zero_indices)

    return oracle


def build_grover_circuit(num_qubits: int, marked_states: List[str], max_iterations: int) -> QuantumCircuit:
    """Construct the Grover circuit with a capped number of iterations."""
    oracle = grover_oracle(marked_states, num_qubits)
    grover_op = GroverOperator(oracle, insert_barriers=True)

    amplitude = math.sqrt(len(marked_states) / (2**num_qubits))
    optimal_iters = int(math.floor(math.pi / (4 * math.asin(amplitude))))
    iterations = min(max_iterations, max(1, optimal_iters))

    circuit = QuantumCircuit(num_qubits)
    circuit.h(range(num_qubits))
    circuit.compose(grover_op.power(iterations), inplace=True)
    circuit.measure_all()
    return circuit


def simulate(
    circuit: QuantumCircuit,
    shots: int,
    optimization_level: int,
) -> dict[str, int]:
    """Transpile with a preset pass manager and execute on the Aer MPS backend."""
    backend = AerSimulator(method="matrix_product_state")
    pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=optimization_level)
    transpiled = pass_manager.run(circuit)

    if RuntimeSampler is not None:
        sampler = RuntimeSampler(mode=backend)
        sampler.options.default_shots = shots
        job = sampler.run([transpiled])
        runtime_result = job.result()
        quasi_counts = runtime_result[0].data.meas.get_counts()
        return {bitstring: int(value) for bitstring, value in quasi_counts.items()}

    sampler = BackendSampler(backend=backend, options={"shots": shots})
    job = sampler.run([transpiled])
    primitive_result = job.result()

    if hasattr(primitive_result, "quasi_dists"):
        distribution = primitive_result.quasi_dists[0]
        return {bitstring: int(round(prob * shots)) for bitstring, prob in distribution.items()}

    return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=int, default=50, help="Number of qubits used by Grover search.")
    parser.add_argument(
        "--marked",
        nargs="+",
        default=None,
        help="List of marked basis states. Defaults to all-zeros and all-ones.",
    )
    parser.add_argument("--max-iterations", type=int, default=10, help="Cap on Grover iterations.")
    parser.add_argument("--shots", type=int, default=10_000, help="Number of samples for the sampler.")
    parser.add_argument("--optimization-level", type=int, default=2, choices=(0, 1, 2, 3), help="Preset PM level.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_marked = ["0" * args.qubits, "1" * args.qubits]
    marked_states = args.marked if args.marked is not None else default_marked
    marked_states = [state.zfill(args.qubits) for state in marked_states]

    circuit = build_grover_circuit(
        num_qubits=args.qubits,
        marked_states=marked_states,
        max_iterations=args.max_iterations,
    )
    counts = simulate(circuit, shots=args.shots, optimization_level=args.optimization_level)

    sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    top_entries = sorted_counts[:5]
    print("Top measurement outcomes:")
    for bitstring, frequency in top_entries:
        print(f"  {bitstring}: {frequency}")

    try:
        from qiskit.visualization import plot_distribution

        figure = plot_distribution(counts, title="Grover search output distribution")
        figure.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
