# [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] def load_results(path):
# [REVIEWED] # [REVIEWED] # [REVIEWED]     with open(path, "r", encoding="utf-8") as f:
# [REVIEWED] # [REVIEWED] # [REVIEWED]         return json.load(f)
# [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] def main():
# [REVIEWED] # [REVIEWED] # [REVIEWED]     parser = argparse.ArgumentParser(description="Evaluate top QAOA hyperparameter candidates.")
# [REVIEWED] # [REVIEWED] # [REVIEWED]     parser.add_argument("--in", dest="infile", required=True, help="Path to JSON results.")
# [REVIEWED] # [REVIEWED] # [REVIEWED]     parser.add_argument("--topk", type=int, default=5, help="Number of candidates to evaluate.")
# [REVIEWED] # [REVIEWED] # [REVIEWED]     parser.add_argument("--epochs", type=int, default=3, help="Epochs for the training stub.")
# [REVIEWED] # [REVIEWED] # [REVIEWED]     parser.add_argument(
# [REVIEWED] # [REVIEWED] # [REVIEWED]         "--trainer",
# [REVIEWED] # [REVIEWED] # [REVIEWED]         default="scripts/train_small_model.py",
# [REVIEWED] # [REVIEWED] # [REVIEWED]         help="Training script to invoke for each candidate.",
    )
    args = parser.parse_args()

    data = load_results(args.infile)
    counts = data.get("counts")
    if not counts:
        print("No counts found in results JSON.", file=sys.stderr)
        return 1

    energies = sorted(counts.items(), key=lambda kv: -kv[1])
    top = energies[: args.topk]

    print("Evaluating top candidates:")
    for bitstring, prob in top:
        print(f"- config={bitstring} prob={prob}")
        cmd = [
            sys.executable,
            args.trainer,
            "--config",
            bitstring,
            "--probability",
            str(prob),
            "--epochs",
            str(args.epochs),
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"Trainer failed for {bitstring}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
