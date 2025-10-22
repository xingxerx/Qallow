#!/usr/bin/env python3
"""
Hybrid quantum learning algorithm for Qallow.

This module implements a lightweight variational quantum classifier that can be
trained on classical data while keeping a fully quantum policy circuit. It
supports mini-batch gradient descent with parameter-shift differentiation and
persistent checkpoints compatible with Qallow's adaptive learning loop.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

try:
    from qiskit import QuantumCircuit, transpile
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "qiskit is required for python.quantum.hybrid_meta_learner. "
        "Install it via 'pip install qiskit qiskit-aer'."
    ) from exc

try:
    from qiskit_aer import AerSimulator
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "qiskit-aer is required for python.quantum.hybrid_meta_learner. "
        "Install it via 'pip install qiskit-aer'."
    ) from exc


ExampleSample = Tuple[Sequence[float], int]


@dataclass
class TrainingEpoch:
    """Container for training diagnostics."""

    epoch: int
    loss: float
    accuracy: float
    learning_rate: float


class HybridQuantumLearner:
    """Variational quantum classifier driven by parameter-shift updates."""

    def __init__(
        self,
        num_qubits: int = 3,
        layers: int = 2,
        learning_rate: float = 0.15,
        shots: int = 2048,
        seed: int | None = None,
    ) -> None:
        if num_qubits < 1:
            raise ValueError("num_qubits must be >= 1")
        if layers < 1:
            raise ValueError("layers must be >= 1")

        self.num_qubits = num_qubits
        self.layers = layers
        self.learning_rate = learning_rate
        self.shots = shots
        self.seed = seed

        rng = np.random.default_rng(seed)
        total_params = layers * num_qubits
        self._params = rng.uniform(-math.pi, math.pi, size=total_params)

        backend_options = {}
        if seed is not None:
            backend_options["seed_simulator"] = seed
            backend_options["seed_transpiler"] = seed
        self._backend = AerSimulator(**backend_options)

        self._history: List[TrainingEpoch] = []

    # ------------------------------------------------------------------ helpers
    def _normalise_features(self, features: Sequence[float]) -> List[float]:
        if not features:
            raise ValueError("Feature vector may not be empty.")

        values = [float(val) for val in features]
        if len(values) < self.num_qubits:
            pad_value = values[-1]
            values.extend(pad_value for _ in range(self.num_qubits - len(values)))
        elif len(values) > self.num_qubits:
            values = values[: self.num_qubits]

        clipped = [max(0.0, min(1.0, val)) for val in values]
        return clipped

    def _build_circuit(self, norm_features: Sequence[float], params: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, self.num_qubits)

        for idx, feature in enumerate(norm_features):
            qc.ry(feature * math.pi, idx)

        param_idx = 0
        for _layer in range(self.layers):
            for qubit in range(self.num_qubits):
                qc.ry(params[param_idx], qubit)
                param_idx += 1

            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)

        qc.barrier()
        qc.measure_all()
        return qc

    def _expectation(self, features: Sequence[float], params: np.ndarray) -> float:
        norm_features = self._normalise_features(features)
        circuit = self._build_circuit(norm_features, params)
        compiled = transpile(circuit, self._backend, seed_transpiler=self.seed)
        job = self._backend.run(compiled, shots=self.shots, seed_simulator=self.seed)
        counts = job.result().get_counts()

        expectation = 0.0
        for bitstring, count in counts.items():
            qubit0 = bitstring[-1]  # little-endian ordering
            contribution = 1.0 if qubit0 == "0" else -1.0
            expectation += contribution * (count / self.shots)
        return expectation

    def _probability(self, expectation: float) -> float:
        prob = 0.5 * (1.0 + expectation)
        return max(1e-6, min(1.0 - 1e-6, prob))

    def _loss_for_batch(self, batch: Iterable[ExampleSample], params: np.ndarray) -> float:
        total_loss = 0.0
        batch_list = list(batch)
        if not batch_list:
            return 0.0

        for features, label in batch_list:
            expectation = self._expectation(features, params)
            prob = self._probability(expectation)
            if label not in (0, 1):
                raise ValueError(f"Labels must be 0 or 1. Got {label!r}.")
            total_loss += -(
                label * math.log(prob) + (1 - label) * math.log(1.0 - prob)
            )

        return total_loss / len(batch_list)

    def _accuracy(self, dataset: Iterable[ExampleSample], params: np.ndarray) -> float:
        dataset_list = list(dataset)
        if not dataset_list:
            return 0.0

        correct = 0
        for features, label in dataset_list:
            expectation = self._expectation(features, params)
            prob = self._probability(expectation)
            prediction = 1 if prob >= 0.5 else 0
            if prediction == int(label):
                correct += 1
        return correct / len(dataset_list)

    def _parameter_shift_gradient(
        self, batch: Iterable[ExampleSample], params: np.ndarray
    ) -> np.ndarray:
        gradients = np.zeros_like(params)
        shift = math.pi / 2.0
        for idx in range(len(params)):
            plus_params = params.copy()
            minus_params = params.copy()
            plus_params[idx] += shift
            minus_params[idx] -= shift

            loss_plus = self._loss_for_batch(batch, plus_params)
            loss_minus = self._loss_for_batch(batch, minus_params)

            gradients[idx] = 0.5 * (loss_plus - loss_minus)
        return gradients

    # ---------------------------------------------------------------- interface
    @property
    def parameters(self) -> np.ndarray:
        return self._params.copy()

    @property
    def history(self) -> List[TrainingEpoch]:
        return list(self._history)

    def predict_proba(self, features: Sequence[float]) -> float:
        expectation = self._expectation(features, self._params)
        return self._probability(expectation)

    def predict_label(self, features: Sequence[float]) -> int:
        prob = self.predict_proba(features)
        return 1 if prob >= 0.5 else 0

    def evaluate(self, dataset: Iterable[ExampleSample]) -> Tuple[float, float]:
        dataset_list = list(dataset)
        if not dataset_list:
            raise ValueError("Dataset is empty; cannot evaluate.")
        loss = self._loss_for_batch(dataset_list, self._params)
        acc = self._accuracy(dataset_list, self._params)
        return loss, acc

    def train(
        self,
        dataset: Iterable[ExampleSample],
        epochs: int = 10,
        batch_size: int | None = None,
        shuffle: bool = True,
        patience: int | None = None,
    ) -> List[TrainingEpoch]:
        samples = list(dataset)
        if not samples:
            raise ValueError("Training dataset is empty.")

        batch_size = batch_size or len(samples)
        best_loss = float("inf")
        stagnation = 0

        for epoch in range(1, epochs + 1):
            if shuffle:
                random.Random(self.seed + epoch if self.seed is not None else None).shuffle(samples)

            for start in range(0, len(samples), batch_size):
                batch = samples[start : start + batch_size]
                gradient = self._parameter_shift_gradient(batch, self._params)
                self._params -= self.learning_rate * gradient

            loss = self._loss_for_batch(samples, self._params)
            acc = self._accuracy(samples, self._params)

            record = TrainingEpoch(
                epoch=epoch,
                loss=loss,
                accuracy=acc,
                learning_rate=self.learning_rate,
            )
            self._history.append(record)

            if loss + 1e-6 < best_loss:
                best_loss = loss
                stagnation = 0
            else:
                stagnation += 1

            if patience is not None and stagnation >= patience:
                break

        return self.history

    # ---------------------------------------------------------------- persistence
    def save_state(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "num_qubits": self.num_qubits,
            "layers": self.layers,
            "learning_rate": self.learning_rate,
            "shots": self.shots,
            "seed": self.seed,
            "params": self._params.tolist(),
            "history": [asdict(epoch) for epoch in self._history],
        }
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return output_path

    @classmethod
    def load_state(cls, path: str | Path) -> "HybridQuantumLearner":
        input_path = Path(path)
        if not input_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {input_path}")
        with input_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        learner = cls(
            num_qubits=int(payload["num_qubits"]),
            layers=int(payload["layers"]),
            learning_rate=float(payload["learning_rate"]),
            shots=int(payload["shots"]),
            seed=payload.get("seed"),
        )
        learner._params = np.array(payload["params"], dtype=float)
        learner._history = [
            TrainingEpoch(
                epoch=int(item["epoch"]),
                loss=float(item["loss"]),
                accuracy=float(item["accuracy"]),
                learning_rate=float(item["learning_rate"]),
            )
            for item in payload.get("history", [])
        ]
        return learner


__all__ = ["HybridQuantumLearner", "TrainingEpoch", "ExampleSample"]

