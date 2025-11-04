#!/usr/bin/env python3
"""Bridge script between Qallow CPU kernel and Cirq execution layer."""







def _parse_states(raw: str) -> List[int]:
    tokens = []
    for chunk in raw.split(","):
        token = chunk.strip()
        if not token:
            continue
        if token.upper() == "N":
            tokens.append(0)
            continue
        try:
            tokens.append(int(token))
        except ValueError:
            raise ValueError(f"Invalid state token: {token}") from None
    if not tokens:
        tokens = [-1, 0, 1]
    return tokens


def _coherence_from_counts(counts: dict[str, float]) -> float:
    if not counts:
        return 0.0
    coherence = 0.0
    total = 0.0
    for bitstring, probability in counts.items():
        try:
            prob = float(probability)
        except (TypeError, ValueError):
            continue
        if prob < 0.0:
            continue
        if len(bitstring) == 0:
            continue
        ones = bitstring.count("1")
        coherence += prob * (ones / len(bitstring))
        total += prob
    if total <= 0.0:
        return 0.0
    return max(0.0, min(1.0, coherence))


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute Cirq bridge for Qallow coherence phase")
    parser.add_argument("--states", type=str, default="-1,0,1", help="Comma-separated ternary states")
    parser.add_argument("--shots", type=int, default=512, help="Number of repetitions for the circuit")
    args = parser.parse_args()

    ternary_states = _parse_states(args.states)
    result = run_ternary_sim(ternary_states, shots=max(1, args.shots), prefer_hardware=False, require_hardware=False)

    coherence = _coherence_from_counts(dict(result.counts))

    print(f"backend={result.backend_name}")
    print(f"source={result.source}")
    print(f"shots={result.shots}")
    print(f"coherence={coherence:.6f}")
    print(json.dumps({
        "counts": dict(result.counts),
        "backend": result.backend_name,
        "source": result.source,
        "shots": result.shots,
        "states": ternary_states,
        "coherence": coherence,
    }))

    return 0


if __name__ == "__main__":
    sys.exit(main())
