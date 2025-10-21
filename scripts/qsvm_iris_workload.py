#!/usr/bin/env python3
"""
QSVM workload for the Iris dataset.

This script assembles a quantum-enhanced SVM pipeline that mirrors the IBM
Quantum tutorial while integrating with the Qallow workspace conventions:

  * Loads configuration environment variables from the repository-level `.env`.
  * Uses a GPU-accelerated Aer simulator when available (falls back to CPU).
  * Logs progress to stdout and appends structured results to the Qallow log dir.

Run inside the project virtualenv:
    source venv/bin/activate
    ./scripts/qsvm_iris_workload.py

Optional flags:
    --device {GPU,CPU}    # force simulator device (default: GPU with fallback)
    --binary-only         # restrict to Iris classes 0/1 (default: multi-class)
    --reps N              # number of ZZFeatureMap repetitions (default: 2)
    --log-file NAME       # custom results filename within QALLOW_LOG_DIR
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from qiskit.circuit.library import ZZFeatureMap
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import (
    QiskitRuntimeService,
    SamplerV2,
    Session,
)
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import QuantumKernel


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = ROOT_DIR / "data" / "logs"


def load_env() -> None:
  """Populate os.environ with key/value pairs from the repository .env."""
  env_path = ROOT_DIR / ".env"
  if not env_path.exists():
    return
  for line in env_path.read_text().splitlines():
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
      continue
    key, value = stripped.split("=", 1)
    if key and key not in os.environ:
      os.environ[key] = value


def _env_credentials() -> Tuple[Optional[str], Optional[str]]:
  token = os.getenv("IBM_QUANTUM_TOKEN") or os.getenv("QISKIT_IBM_TOKEN")
  instance = os.getenv("IBM_QUANTUM_INSTANCE") or os.getenv("QISKIT_IBM_INSTANCE")
  return token, instance


def get_runtime_service(channel: str) -> QiskitRuntimeService:
  """Obtain a QiskitRuntimeService, saving credentials from environment if necessary."""
  try:
    return QiskitRuntimeService(channel=channel)
  except Exception as exc:
    logging.warning("Failed to load saved IBM Quantum account for channel %s: %s", channel, exc)

  token, instance = _env_credentials()
  if not token:
    raise RuntimeError(
        "No IBM Quantum credentials available. "
        "Set IBM_QUANTUM_TOKEN / IBM_QUANTUM_INSTANCE or save an account via QiskitRuntimeService.save_account."
    )

  resolved_instance = instance
  if not resolved_instance:
    if channel == "ibm_quantum":
      resolved_instance = "ibm-q/open/main"
      logging.info("Instance not provided; defaulting to %s for channel ibm_quantum.", resolved_instance)
    else:
      raise RuntimeError(
          f"IBM_QUANTUM_INSTANCE (or QISKIT_IBM_INSTANCE) required for channel {channel}."
      )

  logging.info("Saving IBM Quantum account from environment variables for channel %s", channel)
  QiskitRuntimeService.save_account(
      channel=channel,
      token=token,
      instance=resolved_instance,
      overwrite=True,
  )
  return QiskitRuntimeService(channel=channel)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--device",
      choices=("GPU", "CPU"),
      default="GPU",
      help="Preferred Aer simulator device (default: GPU, auto-fallback to CPU).",
  )
  parser.add_argument(
      "--binary-only",
      action="store_true",
      help="Train on Iris classes 0 and 1 only (default: all three classes).",
  )
  parser.add_argument(
      "--real-backend",
      action="store_true",
      help="Use an IBM Quantum QPU via Qiskit Runtime instead of local simulation.",
  )
  parser.add_argument(
      "--channel",
      default="ibm_cloud",
      help="IBM Quantum channel for runtime access (default: ibm_cloud).",
  )
  parser.add_argument(
      "--backend-name",
      help="Explicit IBM Quantum backend name (e.g. ibm_torino).",
  )
  parser.add_argument(
      "--reps",
      type=int,
      default=2,
      help="Number of ZZFeatureMap repetitions (default: 2).",
  )
  parser.add_argument(
      "--log-file",
      default="qsvm_iris_results.jsonl",
      help="Results file name written under QALLOW_LOG_DIR.",
  )
  parser.add_argument(
      "--test-size",
      type=float,
      default=0.3,
      help="Fraction of dataset reserved for test split (default: 0.3).",
  )
  parser.add_argument(
      "--random-state",
      type=int,
      default=42,
      help="Random seed used for train/test split.",
  )
  parser.add_argument(
      "--shots",
      type=int,
      default=4000,
      help="Number of shots for runtime sampler executions (default: 4000).",
  )
  parser.add_argument(
      "--resilience-level",
      type=int,
      choices=range(0, 3),
      default=1,
      help="Resilience level for runtime sampler (default: 1).",
  )
  return parser.parse_args()


def configure_logging(log_dir: Path) -> None:
  """Configure console logging and ensure the log directory exists."""
  log_dir.mkdir(parents=True, exist_ok=True)
  logging.basicConfig(
      level=logging.INFO,
      format="%(levelname)s: %(message)s",
  )
  logging.getLogger("qiskit").setLevel(logging.WARNING)


def prepare_dataset(
    binary_only: bool,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Load, filter, scale, and split the Iris dataset."""
  iris = load_iris()
  X = iris.data
  y = iris.target

  label_description = "3-class"
  if binary_only:
    mask = y != 2
    X = X[mask]
    y = y[mask]
    label_description = "binary (setosa vs versicolor)"

  logging.info("Dataset loaded: %s, %d samples, %d features", label_description, len(X), X.shape[1])

  X_train, X_test, y_train, y_test = train_test_split(
      X,
      y,
      test_size=test_size,
      random_state=random_state,
      stratify=y,
  )

  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)

  logging.info(
      "Dataset split: %d train / %d test samples (test_size=%.2f)",
      len(X_train_scaled),
      len(X_test_scaled),
      test_size,
  )

  return X_train_scaled, X_test_scaled, y_train, y_test


def select_backend(preferred_device: str) -> AerSimulator:
  """Instantiate an AerSimulator, preferring GPU and falling back gracefully."""
  method = "statevector"

  if preferred_device == "GPU":
    try:
      backend = AerSimulator(method=method, device="GPU")
      logging.info("Using AerSimulator with GPU acceleration (cuQuantum).")
      return backend
    except Exception as exc:  # pragma: no cover - depends on system configuration
      logging.warning("GPU backend unavailable (%s). Falling back to CPU.", exc)

  backend = AerSimulator(method=method, device="CPU")
  logging.info("Using AerSimulator with CPU execution.")
  return backend


class TrackingSampler:
  """Proxy around SamplerV2 that records submitted job IDs."""

  def __init__(self, sampler: SamplerV2):
    self._sampler = sampler
    self.job_ids: list[str] = []

  def run(self, *args, **kwargs):
    job = self._sampler.run(*args, **kwargs)
    try:
      self.job_ids.append(job.job_id())
    except Exception:  # pragma: no cover - depends on runtime internals
      pass
    return job

  def __getattr__(self, item):
    return getattr(self._sampler, item)

  def close(self):
    if hasattr(self._sampler, "close"):
      self._sampler.close()


def build_runtime_sampler(
    channel: str,
    backend_name: Optional[str],
    shots: int,
    resilience_level: int,
) -> Tuple[TrackingSampler, str, Session]:
  """Create a SamplerV2 session targeting an IBM Quantum backend."""
  service = get_runtime_service(channel)

  if backend_name:
    target_backend = backend_name
    logging.info("Using requested IBM Quantum backend: %s", backend_name)
  else:
    backend_obj = service.least_busy(simulator=False, operational=True)
    target_backend = backend_obj.name
    logging.info("Selected least-busy IBM Quantum backend: %s", target_backend)

  session = Session(service=service, backend=target_backend)
  sampler = SamplerV2(session=session)
  sampler.options.default_shots = shots
  sampler.options.resilience_level = resilience_level

  tracking_sampler = TrackingSampler(sampler)
  return tracking_sampler, target_backend, session


def train_qsvm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    quantum_kernel: QuantumKernel,
) -> QSVC:
  """Train QSVC using the provided quantum kernel."""
  model = QSVC(quantum_kernel=quantum_kernel)
  logging.info("Training QSVC...")
  model.fit(X_train, y_train)
  logging.info("Training complete.")
  return model


def evaluate_model(model: QSVC, X_test: np.ndarray, y_test: np.ndarray) -> dict:
  """Compute accuracy and the classification report for the trained model."""
  y_pred = model.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
  logging.info("Test accuracy: %.4f", accuracy)
  return {
      "accuracy": accuracy,
      "classification_report": report,
  }


def persist_results(log_dir: Path, filename: str, payload: dict) -> Path:
  """Append the run metadata to a JSONL file under the provided directory."""
  log_dir.mkdir(parents=True, exist_ok=True)
  output_path = log_dir / filename
  with output_path.open("a", encoding="utf-8") as handle:
    json.dump(payload, handle)
    handle.write("\n")
  logging.info("Results appended to %s", output_path)
  return output_path


def main() -> int:
  load_env()
  args = parse_args()

  log_dir = Path(os.getenv("QALLOW_LOG_DIR", DEFAULT_LOG_DIR))
  configure_logging(log_dir)

  X_train, X_test, y_train, y_test = prepare_dataset(
      binary_only=args.binary_only,
      test_size=args.test_size,
      random_state=args.random_state,
  )

  feature_map = ZZFeatureMap(
      feature_dimension=X_train.shape[1],
      reps=args.reps,
      entanglement="linear",
  )
  logging.info(
      "Feature map configured: %d qubits, reps=%d, entanglement=linear",
      X_train.shape[1],
      args.reps,
  )

  runtime_session: Optional[Session] = None
  runtime_job_ids: list[str] = []

  if args.real_backend:
    logging.info("Initializing IBM Quantum runtime sampler")
    try:
      sampler, backend_name, runtime_session = build_runtime_sampler(
          channel=args.channel,
          backend_name=args.backend_name,
          shots=args.shots,
          resilience_level=args.resilience_level,
      )
    except Exception as exc:
      logging.error("Failed to initialize IBM Quantum runtime: %s", exc)
      return 1
    quantum_kernel = QuantumKernel(feature_map=feature_map, sampler=sampler)
    backend_descriptor = {
        "name": backend_name,
        "device": "QPU",
        "method": "sampler",
        "shots": args.shots,
        "resilience_level": args.resilience_level,
    }
    session_id = getattr(runtime_session, "session_id", None)
    if session_id:
      backend_descriptor["session_id"] = session_id
      logging.info("Runtime session established: %s", session_id)
  else:
    backend = select_backend(args.device)
    quantum_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=backend)
    backend_descriptor = {
        "name": backend.name() if hasattr(backend, "name") else "AerSimulator",
        "device": getattr(backend.options, "device", "CPU"),
        "method": getattr(backend.options, "method", "statevector"),
    }

  model = train_qsvm(X_train, y_train, quantum_kernel=quantum_kernel)
  metrics = evaluate_model(model, X_test, y_test)

  sampler_obj = getattr(quantum_kernel, "sampler", None)
  if sampler_obj is None:
    sampler_obj = getattr(quantum_kernel, "_sampler", None)
  if args.real_backend and isinstance(sampler_obj, TrackingSampler):
    runtime_job_ids = sampler_obj.job_ids
    logging.info(
        "Runtime sampler submitted %d job(s). Latest job ID: %s",
        len(runtime_job_ids),
        runtime_job_ids[-1] if runtime_job_ids else "n/a",
    )
    logging.info("Track workload progress at https://quantum.cloud.ibm.com/ (My recent workloads).")

  results_payload = {
      "timestamp": datetime.utcnow().isoformat() + "Z",
      "dataset": "iris",
      "binary_only": args.binary_only,
      "feature_map": {
          "type": "ZZFeatureMap",
          "reps": args.reps,
          "entanglement": "linear",
          "feature_dimension": X_train.shape[1],
      },
      "backend": backend_descriptor,
      "metrics": metrics,
      "job_ids": runtime_job_ids,
  }

  persist_results(log_dir, args.log_file, results_payload)

  if isinstance(sampler_obj, TrackingSampler):
    try:
      sampler_obj.close()
    except Exception:  # pragma: no cover - cleanup best-effort
      pass

  if runtime_session is not None:
    try:
      runtime_session.close()
    except Exception:  # pragma: no cover - cleanup best-effort
      pass

  logging.info("QSVM Iris workload completed successfully.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
