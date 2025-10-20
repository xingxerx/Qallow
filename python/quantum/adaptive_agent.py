# -*- coding: utf-8 -*-
"""Quantum-assisted adaptive decision module for Qallow.

This module wraps a lightweight Qiskit policy circuit that can be trained
online against Qallow telemetry. The agent encodes telemetry-derived features
into a parameterised two-qubit circuit, executes it on the chosen backend
(Aer simulator by default), and interprets measurement outcomes as phase
actions. Rewards computed from telemetry deltas are used to adjust the circuit
parameters, producing a very small reinforcement-style learning loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence, Tuple

try:
    from qiskit import QuantumCircuit, transpile
except ImportError as exc:  # pragma: no cover - dependency may be optional during linting
    raise RuntimeError(
        "qiskit is required for python.quantum.adaptive_agent. "
        "Install it via 'pip install qiskit qiskit-aer'."
    ) from exc

try:
    from qiskit_aer import AerSimulator
except ImportError as exc:  # pragma: no cover - allow explicit message
    raise RuntimeError(
        "qiskit-aer is required for python.quantum.adaptive_agent. "
        "Install it via 'pip install qiskit-aer'."
    ) from exc


PHASE_ACTIONS = {
    "00": 14,
    "01": 15,
    "10": 16,
    "11": 16,
}


@dataclass
class QuantumAdaptiveAgent:
    """Minimal adaptive controller backed by a parameterised quantum circuit."""

    shots: int = 512
    learning_rate: float = 0.12
    exploration: float = 0.05
    seed: int | None = None
    _params: Tuple[float, float] = field(default_factory=lambda: (math.pi / 4, math.pi / 6))
    _backend: AerSimulator = field(init=False)

    def __post_init__(self) -> None:
        backend_options = {}
        if self.seed is not None:
            backend_options["seed_simulator"] = self.seed
            backend_options["seed_transpiler"] = self.seed
        self._backend = AerSimulator(**backend_options)

    @property
    def parameters(self) -> Tuple[float, float]:
        return self._params

    def _encode_angles(self, features: Sequence[float]) -> Tuple[float, float]:
        if not features:
            raise ValueError("At least one telemetry feature is required.")

        f0 = float(features[0])
        f1 = float(features[1]) if len(features) > 1 else float(features[0])

        f0 = max(0.0, min(1.0, f0))
        f1 = max(0.0, min(1.0, f1))

        theta0 = f0 * math.pi
        theta1 = f1 * math.pi
        return theta0, theta1

    def _build_circuit(self, features: Sequence[float]) -> QuantumCircuit:
        theta0, theta1 = self._encode_angles(features)
        param0, param1 = self._params

        qc = QuantumCircuit(2, 2)
        qc.ry(theta0 + param0, 0)
        qc.ry(theta1 + param1, 1)
        qc.cz(0, 1)
        qc.ry(param0 * 0.5, 0)
        qc.rz(param1 * 0.5, 1)
        qc.barrier()
        qc.measure(0, 0)
        qc.measure(1, 1)
        return qc

    def choose_action(self, features: Sequence[float]) -> Tuple[int, dict[str, float]]:
        circuit = self._build_circuit(features)
        transpiled = transpile(circuit, self._backend, seed_transpiler=self.seed)
        job = self._backend.run(transpiled, shots=self.shots, seed_simulator=self.seed)
        counts = job.result().get_counts()

        total = float(sum(counts.values())) or 1.0
        probabilities = {state: count / total for state, count in counts.items()}

        best_state = max(probabilities, key=probabilities.get)

        if best_state not in PHASE_ACTIONS:
            best_state = "11"

        action = PHASE_ACTIONS[best_state]

        return action, probabilities

    def update(self, features: Sequence[float], reward: float) -> None:
        reward = max(-1.0, min(1.0, reward))
        _, _ = self._encode_angles(features)
        centre = 0.5
        update0 = (features[0] - centre if features else 0.0)
        update1 = (features[1] - centre if len(features) > 1 else update0)

        new_p0 = self._params[0] + self.learning_rate * reward * update0
        new_p1 = self._params[1] + self.learning_rate * reward * update1

        self._params = (
            max(-math.pi, min(math.pi, new_p0)),
            max(-math.pi, min(math.pi, new_p1)),
        )

        if abs(reward) < self.exploration:
            jitter0 = (self.exploration - abs(reward)) * 0.1
            jitter1 = (self.exploration - abs(reward)) * 0.07
            self._params = (
                max(-math.pi, min(math.pi, self._params[0] + jitter0)),
                max(-math.pi, min(math.pi, self._params[1] - jitter1)),
            )
