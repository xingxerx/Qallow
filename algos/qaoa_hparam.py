# [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] def build_qubo(space):
# [REVIEWED] # [REVIEWED] # [REVIEWED]     """Construct a naive QUBO matrix from the discrete hyperparameter space."""
# [REVIEWED] # [REVIEWED] # [REVIEWED]     keys = list(space.keys())
# [REVIEWED] # [REVIEWED] # [REVIEWED]     options = [space[key]["values"] for key in keys]
# [REVIEWED] # [REVIEWED] # [REVIEWED]     combos = list(itertools.product(*options))
# [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED]     if not combos:
# [REVIEWED] # [REVIEWED] # [REVIEWED]         raise ValueError("No hyperparameter combinations generated from space.")
# [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED]     losses = np.linspace(0.1, 1.0, len(combos))
# [REVIEWED] # [REVIEWED] # [REVIEWED]     qubo = {}
# [REVIEWED] # [REVIEWED] # [REVIEWED]     for i in range(len(combos)):
# [REVIEWED] # [REVIEWED] # [REVIEWED]         for j in range(len(combos)):
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
