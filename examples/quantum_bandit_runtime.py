#!/usr/bin/env python3
"""
Example: hardware-ready VQC bandit policy trained via Qiskit Runtime Sampler.

This script mirrors the classical REINFORCE training loop but executes the
variational circuit on either IBM Quantum hardware or a runtime-compatible
simulator. Circuit evaluations are the cost driver, so epochs default low.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate, RZGate
from qiskit.primitives import BackendSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from torch import optim

EPS = 1e-9


@dataclass
class RuntimeConfig:
    backend_name: Optional[str]
    use_least_busy: bool
    simulator_fallback: bool
    optimization_level: int
    shots: int


@dataclass
class BanditConfig:
    arm_probs: torch.Tensor
    epochs: int
    learning_rate: float
    seed: int
    runtime: RuntimeConfig
    plot: bool


@dataclass
class TrainingResult:
    rewards: List[int]
    actions: List[int]
    parameters: torch.Tensor
    backend_name: str


def _resolve_backend(runtime_cfg: RuntimeConfig) -> tuple[object, str]:
    """Find an IBM Quantum backend or fallback to local Aer if allowed."""
    service = QiskitRuntimeService()

    if runtime_cfg.backend_name:
        backend = service.backend(runtime_cfg.backend_name)
        return backend, backend.name

    if runtime_cfg.use_least_busy:
        backend = service.least_busy(operational=True, simulator=False, min_num_qubits=5)
        return backend, backend.name

    if runtime_cfg.simulator_fallback:
        backend = service.backend("ibmq_qasm_simulator")
        return backend, backend.name

    raise RuntimeError("No backend resolution strategy succeeded.")


def build_policy_circuit(theta: Sequence[float]) -> QuantumCircuit:
    """Construct the two-qubit policy circuit used for action sampling."""
    if len(theta) != 4:
        raise ValueError("Theta must contain exactly four parameters.")

    circuit = QuantumCircuit(2)
    circuit.append(RYGate(theta[0]), [0])
    circuit.append(RYGate(theta[1]), [1])
    circuit.cx(0, 1)
    circuit.append(RZGate(theta[2]), [0])
    circuit.append(RYGate(theta[3]), [1])
    circuit.measure_all()
    return circuit


def sample_policy(
    backend: object,
    theta: Sequence[float],
    shots: int,
    optimization_level: int,
) -> np.ndarray:
    """Compile and execute the policy circuit, returning probabilities."""
    circuit = build_policy_circuit(theta)
    pass_manager = generate_preset_pass_manager(
        backend=backend,
        optimization_level=optimization_level,
    )
    transpiled = pass_manager.run(circuit)

    if isinstance(backend, SamplerV2):  # pragma: no cover
        raise TypeError("Expected backend object, not a Sampler instance.")

    try:
        sampler = SamplerV2(mode=backend)
        sampler.options.default_shots = shots
        job = sampler.run([transpiled])
        result = job.result()
        counts = result[0].data.meas.get_counts()
    except Exception:
        sampler = BackendSampler(backend=backend, options={"shots": shots})
        job = sampler.run([transpiled])
        result = job.result()
        counts = result.quasi_dists[0]
        counts = {bitstring: int(round(prob * shots)) for bitstring, prob in counts.items()}

    probabilities = np.zeros(4, dtype=np.float64)
    for bitstring, count in counts.items():
        index = int(bitstring[::-1], 2)
        probabilities[index] += count

    total_counts = probabilities.sum()
    if total_counts <= 0:
        return np.full(4, 0.25)
    return probabilities / total_counts


def train_bandit(config: BanditConfig) -> TrainingResult:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    backend, backend_name = _resolve_backend(config.runtime)
    theta = torch.full((4,), math.pi / 4, dtype=torch.float32, requires_grad=True)
    optimizer = optim.Adam([theta], lr=config.learning_rate)

    rewards: List[int] = []
    actions: List[int] = []

    for epoch in range(config.epochs):
        probabilities = sample_policy(
            backend=backend,
            theta=theta.detach().cpu().numpy(),
            shots=config.runtime.shots,
            optimization_level=config.runtime.optimization_level,
        )
        action = int(np.random.choice(len(config.arm_probs), p=probabilities))
        reward = int(np.random.rand() < config.arm_probs[action].item())

        optimizer.zero_grad()
        prob_tensor = torch.tensor(probabilities[action], dtype=torch.float32)
        log_prob = torch.log(prob_tensor + EPS)
        loss = -log_prob * reward
        loss.backward()
        optimizer.step()

        rewards.append(reward)
        actions.append(action)
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "probabilities": probabilities.tolist(),
                    "action": action,
                    "reward": reward,
                }
            )
        )

    return TrainingResult(
        rewards=rewards,
        actions=actions,
        parameters=theta.detach().clone(),
        backend_name=backend_name,
    )


def cumulative_mean(values: Sequence[int]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return np.cumsum(arr) / (np.arange(arr.size) + 1)


def plot_training(result: TrainingResult, epochs: int) -> None:
    cumulative_rewards = cumulative_mean(result.rewards)
    fig, (ax_avg, ax_hist) = plt.subplots(1, 2, figsize=(12, 5))

    ax_avg.plot(np.arange(epochs), cumulative_rewards)
    ax_avg.set_title("Cumulative Average Reward")
    ax_avg.set_xlabel("Epoch")
    ax_avg.set_ylabel("Average reward")

    ax_hist.hist(result.actions, bins=np.arange(5) - 0.5, density=True, edgecolor="black")
    ax_hist.set_xticks(range(4))
    ax_hist.set_xlabel("Arm")
    ax_hist.set_ylabel("Selection frequency")
    ax_hist.set_title("Action histogram")

    fig.suptitle(f"Backend: {result.backend_name}")
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
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed.")
    parser.add_argument("--shots", type=int, default=1024, help="Number of measurements per circuit execution.")
    parser.add_argument("--optimization-level", type=int, default=1, choices=(0, 1, 2, 3), help="Preset PM level.")
    parser.add_argument("--backend-name", help="Explicit IBM Quantum backend to use.")
    parser.add_argument("--least-busy", action="store_true", help="Pick the least-busy hardware backend.")
    parser.add_argument(
        "--simulator-fallback",
        action="store_true",
        help="Fallback to ibmq_qasm_simulator when hardware selection fails.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Disable matplotlib plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arm_probs_tensor = torch.tensor(args.arm_probs, dtype=torch.float32)

    runtime_cfg = RuntimeConfig(
        backend_name=args.backend_name,
        use_least_busy=args.least_busy,
        simulator_fallback=args.simulator_fallback,
        optimization_level=args.optimization_level,
        shots=args.shots,
    )
    config = BanditConfig(
        arm_probs=arm_probs_tensor,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        runtime=runtime_cfg,
        plot=not args.no_plot,
    )

    result = train_bandit(config)
    average_reward = float(np.mean(result.rewards))
    print(f"Average reward over {config.epochs} epochs: {average_reward:.3f}")
    print(f"Final parameters: {result.parameters.numpy()}")
    print(f"IBM Quantum backend: {result.backend_name}")

    if config.plot:
        plot_training(result, config.epochs)


if __name__ == "__main__":
    main()
