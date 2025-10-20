import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import yaml


def build_qubo(space):
    """Construct a naive QUBO matrix from the discrete hyperparameter space."""
    keys = list(space.keys())
    options = [space[key]["values"] for key in keys]
    combos = list(itertools.product(*options))

    if not combos:
        raise ValueError("No hyperparameter combinations generated from space.")

    losses = np.linspace(0.1, 1.0, len(combos))
    qubo = {}
    for i in range(len(combos)):
        for j in range(len(combos)):
            qubo[(i, j)] = float((losses[i] + losses[j]) / 2.0)
    return qubo, combos


def main():
    parser = argparse.ArgumentParser(description="Build QAOA QUBO from hyperparameter space.")
    parser.add_argument("--space", required=True, help="Path to YAML hyperparameter space.")
    parser.add_argument("--out", required=True, help="Output path for generated QUBO JSON.")
    args = parser.parse_args()

    with open(args.space, "r", encoding="utf-8") as f:
        space = yaml.safe_load(f)

    qubo, combos = build_qubo(space)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "hyperparameters": list(space.keys()),
                "combos": combos,
                "qubo": {f"{i},{j}": value for (i, j), value in qubo.items()},
            },
            f,
            indent=2,
        )

    print(f"QUBO written to {output_path}")


if __name__ == "__main__":
    main()
