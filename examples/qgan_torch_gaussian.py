#!/usr/bin/env python3
"""
Example: quantum generative adversarial network (qGAN) for a 2D Gaussian.

Uses a SamplerQNN-based generator wrapped in TorchConnector and a classical
PyTorch discriminator to learn a discretized multivariate Gaussian surface.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals

try:
    from qiskit.primitives import StatevectorSampler

    SAMPLER_CLASS = StatevectorSampler
except ImportError:  # pragma: no cover
    from qiskit.primitives import Sampler as SAMPLER_CLASS  # type: ignore[assignment]


EPS = 1e-9


@dataclass
class GaussianDataset:
    grid_points: torch.Tensor
    target_probabilities: torch.Tensor
    coords: np.ndarray
    grid_size: int


@dataclass
class TrainingHistory:
    generator_loss: List[float]
    discriminator_loss: List[float]
    kl_divergence: List[float]


def prepare_gaussian_dataset(
    grid_size: int,
    bounding_box: float,
    mean: Tuple[float, float],
    covariance: Tuple[Tuple[float, float], Tuple[float, float]],
) -> GaussianDataset:
    coords = np.linspace(-bounding_box, bounding_box, grid_size, dtype=np.float64)
    mesh_x, mesh_y = np.meshgrid(coords, coords)
    stacked = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])

    cov_matrix = np.array(covariance, dtype=np.float64)
    inv_cov = np.linalg.inv(cov_matrix)
    det_cov = np.linalg.det(cov_matrix)
    norm_const = 1.0 / (2 * math.pi * math.sqrt(det_cov))

    diff = stacked - np.array(mean, dtype=np.float64)
    exponents = -0.5 * np.einsum("ni,ij,nj->n", diff, inv_cov, diff)
    pdf = norm_const * np.exp(exponents)
    pdf /= pdf.sum()

    grid_tensor = torch.tensor(stacked, dtype=torch.float32)
    probs_tensor = torch.tensor(pdf, dtype=torch.float32).unsqueeze(1)

    return GaussianDataset(
        grid_points=grid_tensor,
        target_probabilities=probs_tensor,
        coords=coords,
        grid_size=grid_size,
    )


def build_generator(
    num_qubits: int,
    ansatz_reps: int,
    sampler_shots: int | None,
    seed: int,
) -> TorchConnector:
    algorithm_globals.random_seed = seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    circuit = QuantumCircuit(num_qubits)
    circuit.h(range(num_qubits))
    ansatz = EfficientSU2(num_qubits, reps=ansatz_reps, entanglement="full")
    circuit.compose(ansatz, inplace=True)

    sampler_kwargs = {}
    if sampler_shots is not None and SAMPLER_CLASS.__name__ != "StatevectorSampler":
        sampler_kwargs["shots"] = sampler_shots
    sampler = SAMPLER_CLASS(**sampler_kwargs) if sampler_kwargs else SAMPLER_CLASS()

    qnn = SamplerQNN(
        circuit=circuit,
        sampler=sampler,
        input_params=[],
        weight_params=list(circuit.parameters),
        sparse=False,
    )
    initial_weights = algorithm_globals.random.random(circuit.num_parameters)
    return TorchConnector(qnn, initial_weights=initial_weights.astype(np.float32))


class Discriminator(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def train_qgan(
    generator: TorchConnector,
    discriminator: Discriminator,
    dataset: GaussianDataset,
    epochs: int,
    learning_rate: float,
    betas: Tuple[float, float],
    weight_decay: float,
    log_interval: int,
) -> TrainingHistory:
    gen_opt = Adam(generator.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)
    disc_opt = Adam(discriminator.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)

    history = TrainingHistory([], [], [])
    targets = dataset.target_probabilities
    points = dataset.grid_points

    start = time.time()
    for epoch in range(epochs):
        discriminator.requires_grad_(False)
        gen_opt.zero_grad()

        generated_probs = generator().reshape(-1, 1)
        generated_probs = torch.clamp(generated_probs, min=EPS)
        generated_probs = generated_probs / generated_probs.sum()

        disc_eval = discriminator(points).detach()
        disc_eval = torch.clamp(disc_eval, min=EPS, max=1 - EPS)

        gen_loss = -torch.sum(generated_probs * torch.log(disc_eval))
        gen_loss.backward()
        gen_opt.step()
        discriminator.requires_grad_(True)

        disc_opt.zero_grad()
        with torch.no_grad():
            current_gen = generator().reshape(-1, 1)
            current_gen = torch.clamp(current_gen, min=EPS)
            current_gen = current_gen / current_gen.sum()

        disc_outputs = discriminator(points)
        disc_outputs = torch.clamp(disc_outputs, min=EPS, max=1 - EPS)

        real_loss = -torch.sum(targets * torch.log(disc_outputs))
        fake_loss = -torch.sum(current_gen * torch.log(1 - disc_outputs))
        disc_loss = real_loss + fake_loss
        disc_loss.backward()
        disc_opt.step()

        with torch.no_grad():
            kl_div = torch.sum(targets * (torch.log(targets + EPS) - torch.log(current_gen + EPS))).item()

        history.generator_loss.append(gen_loss.item())
        history.discriminator_loss.append(disc_loss.item())
        history.kl_divergence.append(kl_div)

        if epoch % log_interval == 0 or epoch == epochs - 1:
            elapsed = time.time() - start
            print(
                f"Epoch {epoch:03d} | KL={kl_div:.4f} | "
                f"G_loss={gen_loss.item():.4f} | D_loss={disc_loss.item():.4f} | "
                f"{elapsed:.1f}s"
            )
            start = time.time()

    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=8, help="Number of discrete values per axis.")
    parser.add_argument("--bounding-box", type=float, default=2.0, help="Grid extent on each axis.")
    parser.add_argument("--ansatz-reps", type=int, default=6, help="EfficientSU2 repetition depth.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Optimizer learning rate.")
    parser.add_argument("--beta1", type=float, default=0.7, help="Adam beta1.")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta2.")
    parser.add_argument("--weight-decay", type=float, default=0.005, help="Adam weight decay.")
    parser.add_argument("--shots", type=int, help="Sampler shots (ignored for StatevectorSampler).")
    parser.add_argument("--seed", type=int, default=123456, help="Random seed.")
    parser.add_argument("--log-interval", type=int, default=5, help="Epoch interval for logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = prepare_gaussian_dataset(
        grid_size=args.grid_size,
        bounding_box=args.bounding_box,
        mean=(0.0, 0.0),
        covariance=((1.0, 0.0), (0.0, 1.0)),
    )

    num_qubits = int(math.log2(dataset.grid_points.shape[0]))
    generator = build_generator(
        num_qubits=num_qubits,
        ansatz_reps=args.ansatz_reps,
        sampler_shots=args.shots,
        seed=args.seed,
    )
    discriminator = Discriminator(input_dim=dataset.grid_points.shape[1])

    history = train_qgan(
        generator=generator,
        discriminator=discriminator,
        dataset=dataset,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        log_interval=args.log_interval,
    )

    with torch.no_grad():
        learned = generator().reshape(-1, 1)
        learned = torch.clamp(learned, min=EPS)
        learned = learned / learned.sum()

    final_kl = history.kl_divergence[-1]
    l1_distance = torch.sum(torch.abs(learned - dataset.target_probabilities)).item()
    max_diff = torch.max(torch.abs(learned - dataset.target_probabilities)).item()

    print("\nTraining summary:")
    print(f"  Final KL divergence: {final_kl:.4f}")
    print(f"  L1 distance: {l1_distance:.4f}")
    print(f"  Max probability diff: {max_diff:.4f}")


if __name__ == "__main__":
    main()
