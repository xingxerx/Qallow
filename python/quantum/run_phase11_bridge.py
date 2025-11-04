# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] #!/usr/bin/env python3
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] """CLI entry point to execute Phase 11 ternary coherence checks via Cirq bridge."""
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] from typing import List
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] from . import run_ternary_sim
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] def parse_states(raw: str) -> List[int]:
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]     values = []
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]     for chunk in raw.split(","):
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]         chunk = chunk.strip()
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]         if not chunk:
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]             continue
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]         values.append(int(chunk))
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]     if not values:
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]         values = [-1, 0, 1]
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
        prefer_hardware=True,
        require_hardware=args.hardware_only,
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
