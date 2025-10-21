#!/usr/bin/env python3
"""
Example: train a quantum support vector classifier on a synthetic dataset.

This demo mirrors the canonical Qiskit Machine Learning workflow:
  * generate and scale classical data,
  * encode the data into quantum states with a ZZFeatureMap,
  * build a QuantumKernel powered by the Aer statevector simulator, and
  * train / evaluate a QSVC model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from qiskit.circuit.library import ZZFeatureMap
from qiskit_aer import AerSimulator
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import QuantumKernel


@dataclass
class Dataset:
    train_data: np.ndarray
    test_data: np.ndarray
    train_labels: np.ndarray
    test_labels: np.ndarray


def generate_dataset(
    n_samples: int,
    n_features: int,
    test_size: float,
    random_seed: int,
) -> Dataset:
    """Generate and split a synthetic binary classification dataset."""
    features, labels = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_repeated=0,
        n_clusters_per_class=1,
        class_sep=1.5,
        random_state=random_seed,
    )

    train_data, test_data, train_labels, test_labels = train_test_split(
        features,
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


def build_quantum_kernel(num_features: int, reps: int) -> QuantumKernel:
    """Return a QuantumKernel using a ZZFeatureMap and Aer statevector simulator."""
    feature_map = ZZFeatureMap(feature_dimension=num_features, reps=reps)
    backend = AerSimulator(method="statevector")
    return QuantumKernel(feature_map=feature_map, quantum_instance=backend)


def train_qsvc(train_data: np.ndarray, train_labels: np.ndarray, kernel: QuantumKernel) -> QSVC:
    """Fit a QSVC model with the provided quantum kernel."""
    model = QSVC(quantum_kernel=kernel)
    model.fit(train_data, train_labels)
    return model


def evaluate_model(model: QSVC, test_data: np.ndarray, test_labels: np.ndarray) -> Tuple[float, np.ndarray]:
    """Return the accuracy and prediction vector for the QSVC model."""
    predictions = model.predict(test_data)
    accuracy = (predictions == test_labels).mean()
    return accuracy, predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100, help="Number of data points to generate.")
    parser.add_argument("--features", type=int, default=2, help="Number of classical features.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction.")
    parser.add_argument("--reps", type=int, default=2, help="Repetition depth for the ZZFeatureMap.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = generate_dataset(
        n_samples=args.samples,
        n_features=args.features,
        test_size=args.test_size,
        random_seed=args.seed,
    )
    kernel = build_quantum_kernel(num_features=args.features, reps=args.reps)
    model = train_qsvc(dataset.train_data, dataset.train_labels, kernel)
    accuracy, predictions = evaluate_model(model, dataset.test_data, dataset.test_labels)

    print("QSVC configuration:")
    print(f"  samples: {args.samples}")
    print(f"  features: {args.features}")
    print(f"  ZZFeatureMap reps: {args.reps}")
    print()
    print(f"Classification accuracy: {accuracy:.2f}")
    print("Predictions:", predictions)


if __name__ == "__main__":
    main()
