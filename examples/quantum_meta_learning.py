#!/usr/bin/env python3
"""Demonstration of the HybridQuantumLearner on synthetic data."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.quantum import HybridQuantumLearner, ExampleSample  # noqa: E402


def generate_ring_dataset(
    samples: int,
    noise: float = 0.05,
    seed: int | None = None,
) -> List[ExampleSample]:
    """Binary dataset where points outside a ring are labelled 1."""
    if samples <= 0:
        raise ValueError("samples must be > 0")
    rng = np.random.default_rng(seed)

    dataset: List[ExampleSample] = []
    for _ in range(samples):
        radius = rng.uniform(0.05, 1.15)
        angle = rng.uniform(0.0, 2.0 * math.pi)

        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        feature_vector: Sequence[float] = (
            (x + 1.2) / 2.4,  # normalize to [0, 1]
            (y + 1.2) / 2.4,
            min(1.0, radius / 1.15),
        )

        label = 1 if radius > 0.6 else 0
        if rng.random() < noise:
            label = 1 - label

        dataset.append((feature_vector, label))

    return dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the HybridQuantumLearner on a synthetic ring dataset.",
    )
    parser.add_argument("--train-samples", type=int, default=40, help="Number of training samples.")
    parser.add_argument("--test-samples", type=int, default=20, help="Number of evaluation samples.")
    parser.add_argument("--epochs", type=int, default=6, help="Training epochs.")
    parser.add_argument("--layers", type=int, default=2, help="Variational circuit layers.")
    parser.add_argument("--learning-rate", type=float, default=0.15, help="Learning rate.")
    parser.add_argument("--shots", type=int, default=2048, help="Shots for each circuit execution.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility.")
    parser.add_argument(
        "--save-checkpoint",
        type=Path,
        help="Optional path to save the trained learner state (JSON).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    training_set = generate_ring_dataset(args.train_samples, seed=args.seed)
    test_set = generate_ring_dataset(args.test_samples, seed=args.seed + 1)

    learner = HybridQuantumLearner(
        num_qubits=3,
        layers=args.layers,
        learning_rate=args.learning_rate,
        shots=args.shots,
        seed=args.seed,
    )

    history = learner.train(training_set, epochs=args.epochs, batch_size=8, patience=2)

    print("Training diagnostics:")
    for record in history:
        print(
            f"  epoch={record.epoch:02d} loss={record.loss:.4f} "
            f"acc={record.accuracy:.3f} lr={record.learning_rate:.3f}"
        )

    loss, accuracy = learner.evaluate(test_set)
    print(f"\nEvaluation on held-out set: loss={loss:.4f} accuracy={accuracy:.3f}")

    sample_features, expected = test_set[0]
    probability = learner.predict_proba(sample_features)
    predicted = learner.predict_label(sample_features)
    print(
        f"Sample prediction -> expected={expected} predicted={predicted} "
        f"probability={probability:.4f}"
    )

    if args.save_checkpoint:
        path = learner.save_state(args.save_checkpoint)
        print(f"\nCheckpoint written to {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
