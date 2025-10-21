#!/usr/bin/env python3
"""
IBM Quantum workload bootstrapper.

This script reproduces the Bell-state tutorial workflow:
  1. Verify IBM Quantum credentials.
  2. Build a Bell-state circuit and corresponding observables.
  3. Transpile for the requested backend (fake simulator by default).
  4. Execute via the Qiskit Runtime Estimator primitive.
  5. Print the expectation values so the job shows up in the IBM Quantum workspace.

Usage examples:
  # Use the fake simulator (no QPU minutes consumed)
  ./scripts/ibm_quantum_workload.py

  # Target the least-busy real backend (consumes account QPU time)
  ./scripts/ibm_quantum_workload.py --real-backend

  # Target a specific backend by name
  ./scripts/ibm_quantum_workload.py --backend-name ibm_torino
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime import QiskitRuntimeService


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--channel",
      default="ibm_cloud",
      help="IBM Quantum channel to use (e.g. ibm_cloud, ibm_quantum_platform).",
  )
  backend_group = parser.add_mutually_exclusive_group()
  backend_group.add_argument(
      "--real-backend",
      action="store_true",
      help="Use the least-busy real IBM Quantum backend.",
  )
  backend_group.add_argument(
      "--backend-name",
      help="Run on the specified backend (e.g. ibm_torino).",
  )
  parser.add_argument(
      "--shots",
      type=int,
      default=4000,
      help="Number of shots for the estimator primitive (default: 4000).",
  )
  parser.add_argument(
      "--resilience-level",
      type=int,
      choices=range(0, 3),
      default=1,
      help="Error-mitigation resilience level for Estimator (0, 1, or 2).",
  )
  parser.add_argument(
      "--optimization-level",
      type=int,
      choices=range(0, 4),
      default=1,
      help="Transpiler optimization level (0-3).",
  )
  return parser.parse_args()


def load_dotenv():
  env_path = Path(__file__).resolve().parents[1] / ".env"
  if not env_path.exists():
    return
  for line in env_path.read_text().splitlines():
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
      continue
    key, value = stripped.split("=", 1)
    if key and key not in os.environ:
      os.environ[key] = value


def _env_credentials():
  token = os.getenv("IBM_QUANTUM_TOKEN") or os.getenv("QISKIT_IBM_TOKEN")
  instance = os.getenv("IBM_QUANTUM_INSTANCE") or os.getenv("QISKIT_IBM_INSTANCE")
  return token, instance


def _try_service(channel: str, token: str | None, instance: str | None):
  try:
    service = QiskitRuntimeService(channel=channel)
    account = service.active_account()
    if account:
      print(f"[INFO] Connected to IBM Quantum instance ({channel}): {account.get('instance')}")
      return service
  except Exception as exc:
    print(f"[WARN] {exc}")

  if not token:
    return None

  resolved_instance = instance
  if not resolved_instance:
    if channel == "ibm_quantum":
      resolved_instance = "ibm-q/open/main"
    else:
      return None

  print(f"[INFO] Saving IBM Quantum account for channel '{channel}' via environment variables")
  QiskitRuntimeService.save_account(
      channel=channel,
      token=token,
      instance=resolved_instance,
      overwrite=True,
  )
  service = QiskitRuntimeService(channel=channel)
  account = service.active_account()
  if account:
    print(f"[INFO] Connected to IBM Quantum instance ({channel}): {account.get('instance')}")
    return service
  return None


def ensure_service(preferred_channel: str) -> QiskitRuntimeService:
  token, instance = _env_credentials()

  attempted = []
  for channel in dict.fromkeys([preferred_channel, "ibm_quantum", "ibm_cloud", "ibm_quantum_platform"]):
    attempted.append(channel)
    service = _try_service(channel, token, instance)
    if service:
      return service

  raise RuntimeError(
      "Unable to initialize IBM Quantum service. "
      f"Tried channels: {', '.join(attempted)}. "
      "Set IBM_QUANTUM_TOKEN/IBM_QUANTUM_INSTANCE (or QISKIT_IBM_TOKEN / QISKIT_IBM_INSTANCE) "
      "and, if necessary, specify --channel."
  )


def pick_backend(
    service: QiskitRuntimeService,
    backend_name: str | None,
    real_backend: bool,
):
  if backend_name:
    backend = service.backend(backend_name)
    print(f"[INFO] Using requested backend: {backend.name}")
    return backend

  if real_backend:
    backend = service.least_busy(simulator=False, operational=True)
    print(f"[INFO] Selected least-busy hardware backend: {backend.name}")
    return backend

  from qiskit_ibm_runtime.fake_provider import FakeFez

  backend = FakeFez()
  print("[INFO] Using FakeFez simulator backend (no QPU minutes consumed)")
  return backend


def build_bell_program() -> Tuple[QuantumCircuit, List[SparsePauliOp], List[str]]:
  qc = QuantumCircuit(2, name="bell")
  qc.h(0)
  qc.cx(0, 1)

  observable_labels = ["IZ", "IX", "ZI", "XI", "ZZ", "XX"]
  observables = [SparsePauliOp(label) for label in observable_labels]
  return qc, observables, observable_labels


def transpile_for_backend(
    circuit: QuantumCircuit,
    observables: Iterable[SparsePauliOp],
    backend,
    optimization_level: int,
) -> Tuple[QuantumCircuit, List[SparsePauliOp]]:
  pass_manager = generate_preset_pass_manager(
      backend=backend,
      optimization_level=optimization_level,
  )
  isa_circuit = pass_manager.run(circuit)
  layout = getattr(isa_circuit, "layout", None)
  if layout is None:
    mapped_observables = list(observables)
  else:
    mapped_observables = [obs.apply_layout(layout) for obs in observables]

  print(f"[INFO] Transpiled circuit depth: {isa_circuit.depth()}")
  return isa_circuit, mapped_observables


def run_estimator_job(
    backend,
    circuit: QuantumCircuit,
    observables: Iterable[SparsePauliOp],
    shots: int,
    resilience_level: int,
):
  estimator = Estimator(mode=backend)
  estimator.options.default_shots = shots
  estimator.options.resilience_level = resilience_level

  job = estimator.run([(circuit, list(observables))])
  print(f"[INFO] Submitted workload. Job ID: {job.job_id()}")
  return job


def main() -> int:
  load_dotenv()
  args = parse_args()

  try:
    service = ensure_service(args.channel)
  except Exception as exc:
    print(f"[ERROR] {exc}", file=sys.stderr)
    return 1

  backend = pick_backend(service, args.backend_name, args.real_backend)

  circuit, observables, labels = build_bell_program()
  print("[INFO] Bell circuit construction complete")

  transpiled_circuit, mapped_observables = transpile_for_backend(
      circuit,
      observables,
      backend,
      args.optimization_level,
  )

  job = run_estimator_job(
      backend,
      transpiled_circuit,
      mapped_observables,
      args.shots,
      args.resilience_level,
  )

  try:
    result = job.result()
  except Exception as exc:  # pragma: no cover - runtime connectivity
    print(
        "[WARN] Unable to fetch results automatically "
        f"(job is likely queued or network unavailable): {exc}",
        file=sys.stderr,
    )
    print(
        "[INFO] Track the workload on https://quantum.cloud.ibm.com/ "
        f"using Job ID {job.job_id()}",
    )
    return 0

  pub_result = result[0]
  evs = pub_result.data.evs
  stds = pub_result.data.stds

  print("[INFO] Expectation values:")
  for label, ev, std in zip(labels, evs, stds):
    print(f"  {label}: {ev:.4f} ± {std:.4f}")

  print(
      "[INFO] Workload completed. Verify it under 'My recent workloads' on the IBM "
      "Quantum platform."
  )
  return 0


if __name__ == "__main__":
  sys.exit(main())
