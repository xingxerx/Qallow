import argparse
import ctypes
import math
import sys
from pathlib import Path


def load_c_helper(lib_path):
    if not lib_path.exists():
        return None

    try:
        lib = ctypes.CDLL(str(lib_path))
        lib.qallow_qaoa_eval_score.restype = ctypes.c_double
        lib.qallow_qaoa_eval_score.argtypes = [
            ctypes.c_char_p,
            ctypes.c_double,
            ctypes.c_int,
        ]
        return lib
    except OSError as exc:
        print(f"Warning: failed to load {lib_path}: {exc}", file=sys.stderr)
        return None


def python_fallback(bitstring, probability, epochs):
    ones = bitstring.count("1")
    zeros = bitstring.count("0")
    base = probability + ones * 0.05 + zeros * 0.02
    return base * (1.0 + math.log1p(epochs))


def main():
    parser = argparse.ArgumentParser(description="QAOA training stub for candidate evaluation.")
    parser.add_argument("--config", required=True, help="Bitstring representing the candidate.")
    parser.add_argument("--probability", type=float, required=True, help="Selection probability.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of pseudo-training epochs.")
    parser.add_argument(
        "--lib",
        default="build/libqaoa_eval.so",
        help="Path to optional C helper shared library.",
    )
    args = parser.parse_args()

    lib = load_c_helper(Path(args.lib))
    if lib:
        score = lib.qallow_qaoa_eval_score(args.config.encode(), args.probability, args.epochs)
        backend = "C"
    else:
        score = python_fallback(args.config, args.probability, args.epochs)
        backend = "python"

    print(
        f"Training stub ({backend}) config={args.config} epochs={args.epochs} "
        f"prob={args.probability:.4f} score={score:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
