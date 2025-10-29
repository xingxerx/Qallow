#!/usr/bin/env python3
# Minimal QAOA-like scaffold; integrates with Qallow via --file
import argparse, json, math, os, sys
from typing import Dict, Any

try:
    import cirq
except Exception:
    cirq = None

DEFAULT = dict(M=8, b=6, H=4,
               alpha=1.0, beta=1.0, rho=0.1, gamma=5.0, eta=1.0, kappa=0.1, xi=0.1,
               ethics_tau=0.94, layers=2, shots=2048)

def gray2int(g: int) -> int:
    x = g
    while g:
        g >>= 1
        x ^= g
    return x

def make_circuit(M:int, b:int, H:int, layers:int) -> Any:
    if cirq is None:
        return None

    # Create qubits for different registers
    q_mode = cirq.LineQubit.range(M)
    q_ctrl = cirq.LineQubit.range(M, M + b)
    q_eth = cirq.LineQubit.range(M + b, M + b + 2)
    q_mem = cirq.LineQubit.range(M + b + 2, M + b + 2 + b)

    circuit = cirq.Circuit()

    # init (toy)
    for i in range(M):
        circuit.append(cirq.H(q_mode[i]))
    for j in range(b):
        circuit.append(cirq.H(q_ctrl[j]))

    for l in range(layers):
        # ethics projector stub: flip eth[0] if any ctrl bit = 1 then uncompute
        for j in range(b):
            circuit.append(cirq.CNOT(q_ctrl[j], q_eth[0]))
        # cost e^{-iγH}: proxy with RZ on modes + ctrl
        for i in range(M):
            circuit.append(cirq.rz(0.1)(q_mode[i]))
        for j in range(b):
            circuit.append(cirq.rz(0.05)(q_ctrl[j]))
        # ethics-safe mixer proxy: RX conditioned on eth=0
        circuit.append(cirq.X(q_eth[0]))       # eth==0 → 1
        for i in range(M):
            circuit.append(cirq.CXPowGate(exponent=1.0)(q_eth[0], q_mode[i]))
            circuit.append(cirq.rx(0.2)(q_mode[i]))
        for j in range(b):
            circuit.append(cirq.CXPowGate(exponent=1.0)(q_eth[0], q_ctrl[j]))
            circuit.append(cirq.rx(0.2)(q_ctrl[j]))
        circuit.append(cirq.X(q_eth[0]))
        # uncompute flag
        for j in reversed(range(b)):
            circuit.append(cirq.CNOT(q_ctrl[j], q_eth[0]))

    # Add measurements
    circuit.append(cirq.measure(*q_mode, key='mode'))
    circuit.append(cirq.measure(*q_ctrl, key='ctrl'))
    circuit.append(cirq.measure(*q_eth, key='eth'))
    return circuit

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--alg", default="ccc")
    p.add_argument("--config", help="JSON file with params", default=None)
    p.add_argument("--dump-circuit", action="store_true")
    p.add_argument("--export", default="data/logs/ccc_plan.json")
    args, unknown = p.parse_known_args(argv)

    cfg = DEFAULT.copy()
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            cfg.update(json.load(f))

    qc = make_circuit(cfg["M"], cfg["b"], cfg["H"], cfg.get("layers", 2))
    os.makedirs(os.path.dirname(args.export), exist_ok=True)
    plan = dict(alg="ccc", params=cfg, has_cirq=(qc is not None))
    with open(args.export, "w") as f:
        json.dump(plan, f, indent=2)

    if args.dump_circuit:
        print(qc)

    # Print minimal status line for qallow logs
    print("[CCC] Scaffold ready :: export=", args.export, ":: has_qiskit=", qc is not None)
    return 0

if __name__ == "__main__":
    sys.exit(main())
