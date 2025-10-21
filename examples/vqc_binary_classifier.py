#!/usr/bin/env python3
"""
Example: variational quantum classifier for a binary Iris subset.

Workflow:
  * load / scale classical data,
  * encode features via a ZZFeatureMap,
  * build a RealAmplitudes ansatz,
  * train a VQC model with a COBYLA optimizer and Sampler backend.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.primitives import Sampler
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms import VQC


@dataclass
class Dataset:
    train_data: np.ndarray
    test_data: np.ndarray
    train_labels: np.ndarray
    test_labels: np.ndarray


def prepare_binary_iris(
    num_features: int,
    test_size: float,
    random_seed: int,
) -> Dataset:
    """Return a scaled binary Iris dataset restricted to num_features features."""
    iris = load_iris()
    features = iris.data[:100]
    labels = (iris.target[:100] > 0).astype(int)

    train_data, test_data, train_labels, test_labels = train_test_split(
        features[:, :num_features],
        labels,
        test_size=test_size,
        random_state=random_seed,
        stratify=labels,
    )

    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    return Dataset(
        train_data=train_data,
        test_data=test_data,
        train_labels=train_labels,
        test_labels=test_labels,
    )


def build_vqc(
    num_features: int,
    feature_reps: int,
    ansatz_reps: int,
    maxiter: int,
    shots: int,
) -> tuple[VQC, List[float]]:
    """Construct a VQC with tracking callback."""
    feature_map = ZZFeatureMap(feature_dimension=num_features, reps=feature_reps)
    ansatz = RealAmplitudes(num_qubits=num_features, reps=ansatz_reps)
    sampler = Sampler(options={"shots": shots})
    optimizer = COBYLA(maxiter=maxiter)

    loss_history: List[float] = []

    def _callback(weights: np.ndarray, objective_value: float) -> None:
        loss_history.append(objective_value)

    model = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        sampler=sampler,
        callback=_callback,
    )

    return model, loss_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=int, default=2, choices=(2, 3, 4), help="Number of classical features.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--feature-reps", type=int, default=2, help="Repetitions for the ZZFeatureMap.")
    parser.add_argument("--ansatz-reps", type=int, default=3, help="Repetitions for the RealAmplitudes circuit.")
    parser.add_argument("--maxiter", type=int, default=100, help="Maximum COBYLA iterations.")
    parser.add_argument("--shots", type=int, default=1024, help="Sampler shots per circuit evaluation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = prepare_binary_iris(
        num_features=args.features,
        test_size=args.test_size,
        random_seed=args.seed,
    )

    model, loss_history = build_vqc(
        num_features=args.features,
        feature_reps=args.feature_reps,
        ansatz_reps=args.ansatz_reps,
        maxiter=args.maxiter,
        shots=args.shots,
    )

    model.fit(dataset.train_data, dataset.train_labels)
    train_score = model.score(dataset.train_data, dataset.train_labels)
    test_score = model.score(dataset.test_data, dataset.test_labels)

    print("VQC configuration:")
    print(f"  features: {args.features}")
    print(f"  ZZFeatureMap reps: {args.feature_reps}")
    print(f"  RealAmplitudes reps: {args.ansatz_reps}")
    print(f"  COBYLA maxiter: {args.maxiter}")
    print(f"  Sampler shots: {args.shots}")
    print()
    print(f"Train accuracy: {train_score:.2f}")
    print(f"Test accuracy: {test_score:.2f}")
    if loss_history:
        print(f"Final objective value: {loss_history[-1]:.6f}")
        print(f"Recorded {len(loss_history)} loss evaluations during training.")


if __name__ == "__main__":
    main()
