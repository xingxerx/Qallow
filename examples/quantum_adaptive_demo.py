#!/usr/bin/env python3
"""End-to-end adaptive learning demo for Qallow using Qiskit."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.quantum.adaptive_agent import QuantumAdaptiveAgent


TelemetryRow = Dict[str, float]


def read_latest_telemetry(path: Path) -> TelemetryRow:
    if not path.exists():
        raise FileNotFoundError(f"Telemetry CSV not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        latest = None
        for row in reader:
            latest = row
    if latest is None:
        raise RuntimeError(f"No telemetry rows found in {path}")
    parsed: TelemetryRow = {}
    for key, value in latest.items():
        if key == "mode":
            continue
        try:
            parsed[key] = float(value)
        except (ValueError, TypeError):
            pass
    return parsed


def features_from_metrics(metrics: TelemetryRow) -> Tuple[float, float]:
    global_metric = metrics.get("global", 0.5)
    deco = metrics.get("deco", 0.0)
    deco_component = 1.0 - max(0.0, min(1.0, deco))
    return float(global_metric), float(deco_component)


def compute_reward(before: TelemetryRow, after: TelemetryRow) -> float:
    delta_global = after.get("global", 0.0) - before.get("global", 0.0)
    delta_deco = before.get("deco", 0.0) - after.get("deco", 0.0)
    reward = 0.7 * delta_global + 0.3 * delta_deco
    return max(-1.0, min(1.0, reward))


def run_qallow_phase(runner: Path, phase: int, extra_args: Iterable[str]) -> int:
    command = [str(runner), f"--phase={phase}", *extra_args]
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    sys.stdout.write(proc.stdout)
    sys.stdout.flush()
    return proc.returncode


def simulate_step(metrics: TelemetryRow, phase: int) -> TelemetryRow:
    delta_map = {14: 0.006, 15: 0.008, 16: 0.010}
    deco_map = {14: 0.003, 15: 0.004, 16: 0.005}
    delta_global = delta_map.get(phase, 0.005)
    delta_deco = deco_map.get(phase, 0.002)

    updated = dict(metrics)
    updated["global"] = min(0.999, metrics.get("global", 0.5) + delta_global)
    updated["deco"] = max(0.0, metrics.get("deco", 0.0) - delta_deco)
    updated["tick"] = metrics.get("tick", 0.0) + 1.0
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantum adaptive decision demo for Qallow.")
    parser.add_argument("--runner", help="Path to qallow executable (enable real runs instead of simulation).")
    parser.add_argument("--telemetry", default="data/logs/telemetry_stream.csv", help="Telemetry CSV used for feedback.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of adaptive iterations to perform.")
    parser.add_argument("--sleep", type=float, default=1.5, help="Seconds to wait for telemetry refresh after each phase.")
    parser.add_argument("--extra-arg", action="append", default=[], help="Additional CLI arguments passed through to Qallow.")
    parser.add_argument("--simulate", action="store_true", help="Force simulation mode even if a runner is provided.")
    args = parser.parse_args()

    telemetry_path = Path(args.telemetry)
    simulate = args.simulate or not args.runner

    if simulate:
        baseline_metrics = {"tick": 0.0, "global": 0.94, "deco": 0.04}
        print("[demo] Running in simulation mode (no qallow executable provided).")
    else:
        runner_path = Path(args.runner)
        if not runner_path.exists():
            raise FileNotFoundError(f"Runner not found: {runner_path}")
        os.environ.setdefault("QALLOW_QISKIT", "1")
        baseline_metrics = read_latest_telemetry(telemetry_path)
        print(f"[demo] Using runner {runner_path}")

    agent = QuantumAdaptiveAgent(shots=512, learning_rate=0.15, seed=1337)

    for episode in range(1, args.episodes + 1):
        features = features_from_metrics(baseline_metrics)
        action, probabilities = agent.choose_action(features)
        print(f"[demo] Episode {episode}: features={features!r} probabilities={probabilities} -> phase {action}")

        if simulate:
            time.sleep(0.5)
            updated_metrics = simulate_step(baseline_metrics, action)
        else:
            exit_code = run_qallow_phase(runner_path, action, args.extra_arg)
            if exit_code != 0:
                print(f"[demo] Phase {action} exited with {exit_code}; aborting.")
                break
            time.sleep(args.sleep)
            updated_metrics = read_latest_telemetry(telemetry_path)

        reward = compute_reward(baseline_metrics, updated_metrics)
        agent.update(features, reward)
        print(f"[demo] Reward {reward:+.4f} | params={agent.parameters!r}\n")
        baseline_metrics = updated_metrics

    print("[demo] Completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
