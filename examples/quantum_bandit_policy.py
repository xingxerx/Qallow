#!/usr/bin/env python3
"""
Example: two-qubit variational policy gradient for a stochastic bandit.

Implements a shallow VQC with qutip-derived entangling gates and optimizes the
parameters via a REINFORCE-style update using torch.autograd.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
import torch
from torch import optim


EPS = 1e-7


def _ry(theta: torch.Tensor) -> torch.Tensor:
    """Single-qubit RY rotation matrix."""
    half = theta / 2
    c = torch.cos(half)
    s = torch.sin(half)
    return torch.stack(
        (torch.stack((c, -s), dim=-1), torch.stack((s, c), dim=-1)),
        dim=-2,
    ).to(torch.complex64)


def _rz(theta: torch.Tensor) -> torch.Tensor:
    """Single-qubit RZ rotation matrix."""
    half = theta / 2
    phase_pos = torch.exp(1j * half)
    phase_neg = torch.exp(-1j * half)
    return torch.diag(torch.stack((phase_neg, phase_pos))).to(torch.complex64)


def _cnot_matrix() -> torch.Tensor:
    """Return the CNOT matrix sourced from qutip to keep parity with reference code."""
    cnot_qobj = qt.cnot(N=2, control=0, target=1)
    return torch.tensor(cnot_qobj.full(), dtype=torch.complex64)


CNOT = _cnot_matrix()
ID2 = torch.eye(2, dtype=torch.complex64)


def quantum_policy(theta: torch.Tensor) -> torch.Tensor:
    """Return measurement probabilities for the 2-qubit circuit parameterized by theta."""
    if theta.shape != (4,):
        raise ValueError("Theta must be a vector of length 4.")

    state = torch.zeros(4, dtype=torch.complex64)
    state[0] = 1.0 + 0j

    ry0 = _ry(theta[0])
    ry1 = _ry(theta[1])
    rz0 = _rz(theta[2])
    ry2 = _ry(theta[3])

    state = torch.matmul(torch.kron(ry0, ID2), state)
    state = torch.matmul(torch.kron(ID2, ry1), state)
    state = torch.matmul(CNOT, state)
    state = torch.matmul(torch.kron(rz0, ID2), state)
    state = torch.matmul(torch.kron(ID2, ry2), state)

    probabilities = torch.abs(state) ** 2
    probabilities = probabilities / torch.sum(probabilities)
    return probabilities


@dataclass
class BanditConfig:
    arm_probs: torch.Tensor
    epochs: int
    learning_rate: float
    seed: int
    plot: bool


@dataclass
class TrainingResult:
    rewards: List[int]
    actions: List[int]
    parameters: torch.Tensor


def train_bandit(config: BanditConfig) -> TrainingResult:
    """Train the VQC policy on the stochastic bandit."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    theta = torch.full((4,), np.pi / 4, dtype=torch.float32, requires_grad=True)
    optimizer = optim.Adam([theta], lr=config.learning_rate)

    rewards: List[int] = []
    actions: List[int] = []

    for _ in range(config.epochs):
        probs = quantum_policy(theta)
        probs_np = probs.detach().cpu().numpy()
        action = int(np.random.choice(len(config.arm_probs), p=probs_np))

        reward = int(np.random.rand() < config.arm_probs[action].item())

        optimizer.zero_grad()
        log_prob = torch.log(probs[action] + EPS)
        loss = -log_prob * reward
        loss.backward()
        optimizer.step()

        rewards.append(reward)
        actions.append(action)

    return TrainingResult(rewards=rewards, actions=actions, parameters=theta.detach().clone())


def cumulative_mean(values: Sequence[int]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return np.cumsum(arr) / (np.arange(arr.size) + 1)


def plot_training(result: TrainingResult, epochs: int) -> None:
    cum_rewards = cumulative_mean(result.rewards)
    fig, (ax_reward, ax_hist) = plt.subplots(1, 2, figsize=(12, 5))
    ax_reward.plot(np.arange(epochs), cum_rewards)
    ax_reward.set_title("Cumulative Average Reward")
    ax_reward.set_xlabel("Epoch")
    ax_reward.set_ylabel("Average reward")

    ax_hist.hist(result.actions, bins=np.arange(5) - 0.5, density=True, edgecolor="black")
    ax_hist.set_xticks(range(4))
    ax_hist.set_title("Action selection frequency")
    ax_hist.set_xlabel("Arm")
    ax_hist.set_ylabel("Probability")

    fig.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm-probs",
        nargs=4,
        type=float,
        default=[0.1, 0.3, 0.8, 0.2],
        help="Success probabilities for the four bandit arms.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument("--no-plot", action="store_true", help="Disable matplotlib visualizations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arm_probs_tensor = torch.tensor(args.arm_probs, dtype=torch.float32)
    config = BanditConfig(
        arm_probs=arm_probs_tensor,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        plot=not args.no_plot,
    )
    result = train_bandit(config)

    avg_reward = float(np.mean(result.rewards))
    print(f"Average reward over {config.epochs} epochs: {avg_reward:.3f}")
    print("Final parameters:", result.parameters.numpy())

    if config.plot:
        plot_training(result, config.epochs)


if __name__ == "__main__":
    main()
