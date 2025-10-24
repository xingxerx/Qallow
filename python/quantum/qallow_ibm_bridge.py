"""Utilities to bridge Qallow ternary state experiments with Cirq backends."""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import cirq

try:
    import cirq_google
except ImportError:  # pragma: no cover - optional dependency
    cirq_google = None


@dataclass(slots=True)
class TernaryResult:
    """Container for quantum execution results consumed by Qallow telemetry."""

    counts: Mapping[str, float]
    backend_name: str
    shots: int
    source: str  # "hardware" or "simulator"
    logical_counts: Optional[Mapping[str, float]] = None
    metadata: Optional[Mapping[str, float]] = None


def _resolve_surface_code(distance: Optional[int],
                          physical_error_rate: Optional[float]) -> Tuple[int, float]:
    """Resolve surface code parameters using CLI overrides with env/default fallbacks."""

    resolved_distance = distance
    if resolved_distance is None:
        env_distance = os.getenv("QALLOW_SURFACE_CODE_DISTANCE")
        if env_distance:
            try:
                resolved_distance = int(env_distance)
            except ValueError:
                resolved_distance = None
    if resolved_distance is None or resolved_distance < 1:
        resolved_distance = 1

    resolved_error = physical_error_rate
    if resolved_error is None:
        env_error = os.getenv("QALLOW_PHYSICAL_ERROR_RATE")
        if env_error:
            try:
                resolved_error = float(env_error)
            except ValueError:
                resolved_error = None
    if resolved_error is None or resolved_error < 0.0:
        resolved_error = 0.01
    if resolved_error > 1.0:
        resolved_error = 1.0

    return resolved_distance, resolved_error


def _estimate_logical_error_rate(physical_error_rate: float, distance: int) -> float:
    """Estimate logical error rate using a surface-code heuristic."""

    if distance <= 0:
        return max(0.0, min(1.0, physical_error_rate))
    estimate = physical_error_rate ** (distance / 2.0)
    if estimate < 0.0:
        return 0.0
    if estimate > 1.0:
        return 1.0
    return estimate


def _compute_logical_counts(
    counts: Mapping[str, float],
    logical_qubits: int,
    block_size: int,
) -> Optional[Dict[str, float]]:
    """Aggregate physical qubit measurements into logical qubits via majority vote."""

    if block_size <= 1 or logical_qubits <= 0:
        return None

    aggregated: Dict[str, float] = defaultdict(float)
    expected_len = logical_qubits * block_size
    for bitstring, probability in counts.items():
        padded = bitstring.zfill(expected_len)
        logical_bits = []
        for index in range(logical_qubits):
            start = index * block_size
            segment = padded[start : start + block_size]
            ones = segment.count("1")
            zeros = block_size - ones
            logical_bits.append("1" if ones > zeros else "0")
        aggregated["".join(logical_bits)] += probability

    total = sum(aggregated.values())
    if total <= 0.0:
        return dict(aggregated)
    return {key: value / total for key, value in aggregated.items()}


def build_ternary_circuit(
    ternary_states: Sequence[int], *, surface_code_distance: int = 1
) -> cirq.Circuit:
    """Build a Cirq circuit mapping ternary (-1, 0, 1) states onto qubits."""

    logical_states = list(ternary_states) if ternary_states else [0]
    block_size = surface_code_distance * surface_code_distance if surface_code_distance > 1 else 1
    num_qubits = max(len(logical_states) * block_size, 1)
    qubits = [cirq.NamedQubit(f"q{i}") for i in range(num_qubits)]
    operations = []

    for logical_index, state in enumerate(logical_states):
        int_state = max(-1, min(1, int(state)))
        block_start = logical_index * block_size
        anchor_qubit = qubits[block_start]
        for offset in range(block_size):
            qubit = qubits[block_start + offset]
            if int_state > 0:
                operations.append(cirq.X(qubit))
            elif int_state < 0:
                operations.extend([cirq.H(qubit), cirq.Z(qubit), cirq.H(qubit)])
            if block_size > 1 and offset > 0:
                operations.append(cirq.CNOT(anchor_qubit, qubit))

    operations.append(cirq.measure(*qubits, key="m"))
    return cirq.Circuit(operations)


def _resolve_engine_config() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    project_id = os.getenv("QALLOW_CIRQ_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
    processor_id = os.getenv("QALLOW_CIRQ_PROCESSOR_ID")
    endpoint = os.getenv("QALLOW_CIRQ_ENDPOINT")
    return project_id, processor_id, endpoint


def _counts_from_measurements(measurements, shot_count: int) -> Dict[str, float]:
    rows = []
    for sample in measurements:
        rows.append("".join(str(int(bit)) for bit in sample[::-1]))
    raw = Counter(rows)
    if shot_count <= 0:
        return {key: float(value) for key, value in raw.items()}
    return {key: float(value) / float(shot_count) for key, value in raw.items()}


def _execute_on_engine(
    circuit: cirq.Circuit,
    shots: int,
    project_id: str,
    processor_id: str,
    endpoint: Optional[str] = None,
) -> TernaryResult:
    if cirq_google is None:  # pragma: no cover - depends on optional package
        raise RuntimeError("cirq-google is not installed; install it for hardware access.")

    engine_sampler = cirq_google.EngineSampler(
        project_id=project_id,
        processor_id=processor_id,
        endpoint=endpoint,
    )
    result = engine_sampler.run(circuit, repetitions=shots)
    measurements = result.measurements["m"]
    counts = _counts_from_measurements(measurements, shots)

    return TernaryResult(
        counts=dict(counts),
        backend_name=processor_id,
        shots=shots,
        source="hardware",
    )


def _execute_on_simulator(circuit: cirq.Circuit, shots: int) -> TernaryResult:
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=shots)
    measurements = result.measurements["m"]
    counts = _counts_from_measurements(measurements, shots)

    return TernaryResult(
        counts=dict(counts),
        backend_name="cirq_simulator",
        shots=shots,
        source="simulator",
    )


def run_ternary_sim(
    ternary_states: Sequence[int],
    *,
    shots: int = 1024,
    prefer_hardware: bool = True,
    require_hardware: bool = False,
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
    _ = token  # retained for backwards compatibility with earlier Qiskit integration

    logical_states = list(ternary_states)
    if not logical_states:
        logical_states = [0]

    shots = max(1, int(shots))
    surface_code_distance, physical_error_rate = _resolve_surface_code(None, None)
    block_size = (
        surface_code_distance * surface_code_distance if surface_code_distance > 1 else 1
    )
    circuit = build_ternary_circuit(
        logical_states,
        surface_code_distance=surface_code_distance,
    )

    metadata: Dict[str, float] = {
        "surface_code_distance": float(surface_code_distance),
        "physical_error_rate": float(physical_error_rate),
        "logical_error_rate": float(
            _estimate_logical_error_rate(physical_error_rate, surface_code_distance)
        ),
        "qubit_count": float(circuit.num_qubits()),
        "shot_count": float(shots),
    }

    if prefer_hardware:
        project_id, processor_id, endpoint = _resolve_engine_config()
        if project_id and processor_id:
            try:
                result = _execute_on_engine(
                    circuit,
                    shots,
                    project_id=project_id,
                    processor_id=processor_id,
                    endpoint=endpoint,
                )
            except Exception as error:  # pragma: no cover - defensive guard
                if require_hardware:
                    raise RuntimeError(
                        "Cirq hardware execution failed while hardware-only mode is enabled."
                    ) from error
            else:
                logical_counts = _compute_logical_counts(
                    result.counts,
                    logical_qubits=len(logical_states),
                    block_size=block_size,
                )
                result.logical_counts = logical_counts
                result.metadata = dict(metadata)
                return result
        elif require_hardware:
            raise RuntimeError(
                "Hardware execution requested but no Cirq engine configuration was provided."
            )

    if require_hardware:
        raise RuntimeError("Hardware execution was requested but could not be fulfilled.")

    result = _execute_on_simulator(circuit, shots)
    logical_counts = _compute_logical_counts(
        result.counts,
        logical_qubits=len(logical_states),
        block_size=block_size,
    )
    result.logical_counts = logical_counts
    result.metadata = dict(metadata)
    return result


__all__ = [
    "TernaryResult",
    "build_ternary_circuit",
    "run_ternary_sim",
]
