#!/usr/bin/env python3
"""
Simple adaptive updater for Qallow ethics weights.

Uses a synthetic reward signal to tweak weights and thresholds,
then writes the results back to config/*.json for the C runtime.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
WEIGHTS_PATH = CONFIG_DIR / "weights.json"
THRESHOLDS_PATH = CONFIG_DIR / "thresholds.json"
STATE_PATH = ROOT / "adapt_state.json"


def load_json(path: Path, fallback: dict) -> dict:
    if not path.exists():
        return fallback.copy()
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return fallback.copy()


def save_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp.replace(path)


def stabilise(weights: dict) -> dict:
    clipped = {}
    for k, v in weights.items():
        value = float(v)
        if not math.isfinite(value):
            value = 1.0
        value = max(0.8, min(1.2, value))
        clipped[k] = value
    return clipped


def main() -> None:
    weights = load_json(
        WEIGHTS_PATH,
        {"safety_weight": 0.4, "clarity_weight": 0.35, "human_weight": 0.25},
    )
    thresholds = load_json(
        THRESHOLDS_PATH,
        {"min_safety": 0.7, "min_clarity": 0.65, "min_human": 0.6, "min_total": 1.8},
    )
    state = load_json(STATE_PATH, {"previous_reward": 0.0})

    synthetic_reward = random.uniform(-0.2, 0.2)
    previous_reward = float(state.get("previous_reward", 0.0))
    blended = 0.7 * previous_reward + 0.3 * synthetic_reward

    weights = stabilise(weights)
    weights["human_weight"] = max(0.8, min(1.2, weights["human_weight"] + blended * 0.2))
    weights = stabilise(weights)

    thresholds["min_total"] = float(
        min(2.5, max(1.5, thresholds["min_total"] + blended * -0.05))
    )

    save_json(WEIGHTS_PATH, weights)
    save_json(THRESHOLDS_PATH, thresholds)
    save_json(STATE_PATH, {"previous_reward": synthetic_reward})

    print("[train_and_feed] reward={:.3f} -> weights {}".format(synthetic_reward, weights))
    print("[train_and_feed] thresholds updated:", thresholds)


if __name__ == "__main__":
    main()
