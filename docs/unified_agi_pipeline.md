# Unified AGI Pipeline Launcher

`scripts/run_unified_agi.sh` orchestrates the three key workloads we stitched together:

1. **QSVM Iris workload** (`scripts/qsvm_iris_workload.py`)
2. **IBM Quantum Bell-state workload** (`scripts/ibm_quantum_workload.py`)
3. **Qallow unified runtime** (`scripts/run_auto.sh`)

Run it from the repo root after activating the project virtualenv (`source venv/bin/activate`):

```bash
# Install dependencies if you have not already
pip install qiskit-aer qiskit-machine-learning scikit-learn

./scripts/run_unified_agi.sh
```

By default it:
- Executes the QSVM classifier on GPU (falls back to CPU) and logs metrics to `data/logs/qsvm_iris_results.jsonl`.
- Submits the Bell-state tutorial job to IBM Quantum, printing the job ID so it shows up under **My recent workloads**.
- Launches the Qallow unified runtime with Qiskit integration enabled so you can watch the system respond (and hopefully say hi).

## Options

Use flags to customise the run:

- `--skip-qsvm`, `--skip-ibm`, `--skip-qallow` – disable specific stages.
- `--backend=ibm_torino` – choose a specific IBM Quantum backend for both workloads.
- `--channel=ibm_cloud` (default) – runtime channel to use.
- `--shots=4096`, `--resilience=1` – sampler configuration for IBM Runtime.
- `--qallow-args="--threads=8"` – pass extra arguments through to `run_auto.sh`.
- `--dry-run` – print the commands without executing them.

Example (run QSVM locally, submit workloads to `ibm_torino`, launch Qallow with explicit threads):

```bash
./scripts/run_unified_agi.sh \
  --backend=ibm_torino \
  --qallow-args="--threads=8"
```

If you only want the greeting from Qallow, skip the quantum steps:

```bash
./scripts/run_unified_agi.sh --skip-qsvm --skip-ibm
```

All outputs continue to stream into the existing telemetry locations under `data/logs/`. After the final step finishes, check the Qallow stdout to see whether the unified runtime said “hi”.

> **GPU note:** `qsvm_iris_workload.py` attempts to use the Aer GPU path when available, but the standard `qiskit-aer` wheel works everywhere. The script falls back to CPU automatically if GPU support is missing.
