# IBM Quantum Workload Bootstrap

This guide explains how to submit the Bell-state tutorial workload so it appears in your IBM Quantum workspace.

## Prerequisites
- Activate the project virtualenv:
  ```bash
  source venv/bin/activate
  ```
- Ensure Qiskit packages are installed (already present in this repo’s `venv`):
  ```bash
  pip show qiskit qiskit-ibm-runtime matplotlib
  ```
- Save your IBM Quantum credentials once per machine:
  ```python
  from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(
    channel="ibm_cloud",
    token="YOUR_API_TOKEN",
    instance="YOUR_CRN",
    overwrite=True,
)
```

Values placed in `.env` are loaded automatically; supported keys:
- `IBM_QUANTUM_TOKEN` / `IBM_QUANTUM_INSTANCE`
- `QISKIT_IBM_TOKEN` / `QISKIT_IBM_INSTANCE`

If no instance is supplied for the legacy `ibm_quantum` channel the script falls back to `ibm-q/open/main`.

## Submit the Tutorial Workload
```bash
./scripts/ibm_quantum_workload.py
```

The script will:
1. Confirm your IBM Quantum account.
2. Build the Bell-state circuit and observables.
3. Transpile it for the FakeFez simulator (no QPU minutes consumed).
4. Run the Estimator primitive and print expectation values.

When the run finishes, open <https://quantum.cloud.ibm.com/> and check **My recent workloads** for the new entry.

### Run on Real Hardware
```bash
./scripts/ibm_quantum_workload.py --real-backend
```
or specify a backend explicitly:
```bash
./scripts/ibm_quantum_workload.py --backend-name ibm_torino
```

> Hardware runs consume QPU time and may sit in queue; the script prints the job ID so you can track progress in the IBM Quantum dashboard.

## Troubleshooting
- **No IBM Quantum account configured**: Export `IBM_QUANTUM_TOKEN` / `IBM_QUANTUM_INSTANCE` for the run or re-save the account with `QiskitRuntimeService.save_account(...)`.
- **Network errors / job queued**: The script reports the job ID; track it online once connectivity returns or the queue clears.
- **Need plotting**: Copy the expectation values into a notebook and use matplotlib to reproduce tutorial plots.
