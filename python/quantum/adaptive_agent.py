# [REVIEWED] # [REVIEWED] # [REVIEWED] # -*- coding: utf-8 -*-
# [REVIEWED] # [REVIEWED] # [REVIEWED] """Quantum-assisted adaptive decision module for Qallow.
# [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] This module wraps a lightweight Cirq policy circuit that can be trained
# [REVIEWED] # [REVIEWED] # [REVIEWED] online against Qallow telemetry. The agent encodes telemetry-derived features
# [REVIEWED] # [REVIEWED] # [REVIEWED] into a parameterised two-qubit circuit, executes it on a Cirq simulator, and
# [REVIEWED] # [REVIEWED] # [REVIEWED] interprets measurement outcomes as phase actions. Rewards computed from
# [REVIEWED] # [REVIEWED] # [REVIEWED] telemetry deltas are used to adjust the circuit parameters, producing a very
# [REVIEWED] # [REVIEWED] # [REVIEWED] small reinforcement-style learning loop.
# [REVIEWED] # [REVIEWED] # [REVIEWED] """
# [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] from dataclasses import dataclass, field
# [REVIEWED] # [REVIEWED] # [REVIEWED] from typing import Tuple, Dict, Sequence
# [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] try:
# [REVIEWED] # [REVIEWED] # [REVIEWED]     import cirq
# [REVIEWED] # [REVIEWED] # [REVIEWED] except ImportError as exc:  # pragma: no cover - dependency may be optional during linting
# [REVIEWED] # [REVIEWED] # [REVIEWED]     raise RuntimeError(
        "cirq is required for python.quantum.adaptive_agent. "
        "Install it via 'pip install cirq'."
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
    _simulator: cirq.Simulator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._simulator = cirq.Simulator()

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

    def _build_circuit(self, features: Sequence[float]) -> cirq.Circuit:
        theta0, theta1 = self._encode_angles(features)
        param0, param1 = self._params
        qubits = cirq.LineQubit.range(2)
        circuit = cirq.Circuit()

        circuit.append(cirq.ry(theta0 + param0)(qubits[0]))
        circuit.append(cirq.ry(theta1 + param1)(qubits[1]))
        circuit.append(cirq.CZ(qubits[0], qubits[1]))
        circuit.append(cirq.ry(param0 * 0.5)(qubits[0]))
        circuit.append(cirq.rz(param1 * 0.5)(qubits[1]))
        circuit.append(cirq.measure(*qubits, key="m"))

        return circuit

    def choose_action(self, features: Sequence[float]) -> Tuple[int, dict[str, float]]:
        circuit = self._build_circuit(features)
        random_state = self.seed if self.seed is not None else None
        result = self._simulator.run(circuit, repetitions=self.shots, random_state=random_state)
        measurements = result.measurements["m"]
        counts: Dict[str, int] = {}
        for row in measurements:
            bitstring = "".join(str(int(bit)) for bit in row)
            counts[bitstring] = counts.get(bitstring, 0) + 1

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
