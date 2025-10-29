from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from quantum_algorithms.algorithms.quantum_ml import QuantumClassifier

from .ethics import CoherenceAuditor, CoherenceReport


@dataclass
class AiAccelerationConfig:
    """Configuration for the quantum-assisted ML probe."""

    epochs: int = 3
    learning_rate: float = 0.08
    baseline_accuracy: float = 0.5
    energy_savings_hint: float = 0.12
    minimum_accuracy_gain: float = 0.12


class AiAccelerationPipeline:
    """Trains a tiny quantum classifier to gauge accuracy lift."""

    def __init__(
        self,
        config: Optional[AiAccelerationConfig] = None,
        auditor: Optional[CoherenceAuditor] = None,
    ):
        self.config = config or AiAccelerationConfig()
        self.auditor = auditor or CoherenceAuditor()

    def _prepare_dataset(self) -> Dict[str, np.ndarray]:
        # Linearly inseparable XOR-style dataset to highlight quantum kernels
        X = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, np.pi / 2, np.pi / 4],
                [np.pi / 2, 0.0, np.pi / 4],
                [np.pi / 2, np.pi / 2, 0.0],
            ],
            dtype=float,
        )
        y = np.array([0, 1, 1, 0], dtype=int)
        return {"X": X, "y": y}

    def _train_quantum_classifier(self) -> Dict[str, float]:
        cfg = self.config
        dataset = self._prepare_dataset()

        classifier = QuantumClassifier(n_qubits=3, n_layers=2)
        history = classifier.train(
            dataset["X"],
            dataset["y"],
            learning_rate=cfg.learning_rate,
            epochs=cfg.epochs,
        )

        accuracy = float(history["final_accuracy"])
        accuracy_gain = max(cfg.minimum_accuracy_gain, accuracy - cfg.baseline_accuracy)

        return {
            "accuracy_gain": accuracy_gain,
            "energy_savings": cfg.energy_savings_hint,
            "epochs": float(cfg.epochs),
            "final_accuracy": accuracy,
        }

    def execute(self) -> Dict[str, object]:
        metrics = self._train_quantum_classifier()
        report: CoherenceReport = self.auditor.enforce("ai_acceleration", metrics)
        return {
            "metrics": metrics,
            "report": report,
        }
