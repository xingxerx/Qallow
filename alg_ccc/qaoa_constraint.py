#!/usr/bin/env python3
# Minimal QAOA-like scaffold; integrates with Qallow via --file
import argparse, json, math, os, sys
from typing import Dict, Any

try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
except Exception:
    QuantumCircuit = None

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
    if QuantumCircuit is None:
        return None
    q_mode = QuantumRegister(M, "mode")
    q_ctrl = QuantumRegister(b, "ctrl")
    q_eth  = QuantumRegister(2, "eth")
    q_mem  = QuantumRegister(b, "mem")  # single step mirror for demo
    c_out  = ClassicalRegister(M + b + 2, "c")
    qc = QuantumCircuit(q_mode, q_ctrl, q_eth, q_mem, c_out, name="CCC")

    # init (toy)
    for i in range(M):
        qc.h(q_mode[i])
    for j in range(b):
        qc.h(q_ctrl[j])

    for l in range(layers):
        # ethics projector stub: flip eth[0] if any ctrl bit = 1 then uncompute
        for j in range(b):
            qc.cx(q_ctrl[j], q_eth[0])
        # cost e^{-iγH}: proxy with RZ on modes + ctrl
        for i in range(M):
            qc.rz(0.1, q_mode[i])
        for j in range(b):
            qc.rz(0.05, q_ctrl[j])
        # ethics-safe mixer proxy: RX conditioned on eth=0
        qc.x(q_eth[0])       # eth==0 → 1
        for i in range(M):
            qc.crx(0.2, q_eth[0], q_mode[i])
        for j in range(b):
            qc.crx(0.2, q_eth[0], q_ctrl[j])
        qc.x(q_eth[0])
        # uncompute flag
        for j in reversed(range(b)):
            qc.cx(q_ctrl[j], q_eth[0])

    qc.barrier()
    qc.measure(q_mode, c_out[:M])
    qc.measure(q_ctrl, c_out[M:M+b])
    qc.measure(q_eth,  c_out[M+b:M+b+2])
    return qc

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
    plan = dict(alg="ccc", params=cfg, has_qiskit=(qc is not None))
    with open(args.export, "w") as f:
        json.dump(plan, f, indent=2)

    if args.dump_circuit:
        print(qc)

    # Print minimal status line for qallow logs
    print("[CCC] Scaffold ready :: export=", args.export, ":: has_qiskit=", qc is not None)
    return 0

if __name__ == "__main__":
    sys.exit(main())
