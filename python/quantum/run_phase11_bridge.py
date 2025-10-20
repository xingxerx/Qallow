#!/usr/bin/env python3
"""
CLI entry point to execute Phase 11 ternary coherence checks via IBM Quantum bridge.
"""

from __future__ import annotations

import argparse
import json
from typing import List

from . import run_ternary_sim


def parse_states(raw: str) -> List[int]:
    values = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    if not values:
        values = [-1, 0, 1]
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=1024, help="Number of samples to request.")
    parser.add_argument(
        "--states",
        type=str,
        default="-1,0,1",
        help="Comma-separated ternary states emitted by Phase 11.",
    )
    parser.add_argument(
        "--hardware-only",
        action="store_true",
        help="Fail instead of falling back to Aer simulator.",
    )
    args = parser.parse_args()

    shots = max(1, args.shots)
    ternary_states = parse_states(args.states)

    result = run_ternary_sim(
        ternary_states,
        shots=shots,
        prefer_hardware=not args.hardware_only,
    )

    payload = {
        "backend": result.backend_name,
        "source": result.source,
        "shots": result.shots,
        "counts": dict(result.counts),
        "states": ternary_states,
    }

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
