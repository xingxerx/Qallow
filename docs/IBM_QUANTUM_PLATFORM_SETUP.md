# IBM Quantum Platform Setup for Qallow

This guide walks through connecting a Qallow deployment to IBM Quantum services. Follow the three stages in order—account creation, API token registration, and Qiskit integration inside the Qallow workspace.

## 1. Create an IBM Quantum Account
- Visit https://quantum.cloud.ibm.com and choose **Sign Up**. Register with an email address or a supported federated provider such as GitHub.
- Verify the email address, then complete the profile prompt (organization is optional).
- When the dashboard loads confirm that the account status under **Account** → **Usage** reads `Active`.
- Explore the **Systems** tab to review available simulators and hardware. The free tier currently provides roughly 10 minutes of hardware execution time per month with queue-based access; expect 5–30 minute waits during busy periods. Simulators are not metered.

## 2. Generate and Secure an API Token
- In the IBM Quantum dashboard open **Account** → **API Token** and click **Generate new token**.
- Copy the token string and immediately store it in a secure location. Never commit the token to version control.
- Recommended storage options:
  - Export an environment variable in your shell profile:
    ```bash
    export QISKIT_IBM_TOKEN="replace_with_token"
    ```
  - Or persist credentials locally through Qiskit (run once):
    ```python
    from qiskit_ibm_runtime import QiskitRuntimeService

    QiskitRuntimeService.save_account(
        channel="ibm_quantum",
        token="replace_with_token",
        overwrite=True,
    )
    ```
- Regenerate the token if it is ever exposed or if you want to rotate credentials routinely (monthly rotation is recommended).
- Quick validation: launch Python and run `from qiskit_ibm_runtime import QiskitRuntimeService; print(QiskitRuntimeService().backends())`. A populated list confirms the token works.

## 3. Configure Qiskit Inside Qallow
1. **Install dependencies** (Python 3.8+):
   ```bash
   pip install qiskit qiskit-ibm-runtime qiskit-aer
   ```
   If you keep dependencies in a virtual environment for Qallow, activate it first.

2. **Baseline runtime test** using the included example:
   ```bash
   python examples/ibm_quantum_bell.py
   ```
   The script loads the `QISKIT_IBM_TOKEN` environment variable (if present) or reuses saved credentials, selects the least busy hardware backend, and runs a Bell circuit. If hardware access is unavailable it automatically falls back to the Aer simulator. Expect quasi-probabilities near 0.5 for `00` and `11`.

3. **Bridge Qallow Phase 11 workflows to Qiskit** using the helper module in `python/quantum/qallow_ibm_bridge.py`. The module exposes a `run_ternary_sim` function that:
   - Accepts ternary state estimates from Qallow (e.g., `[-1, 0, 1]`).
   - Builds a representative circuit using Qiskit.
   - Submits the job through `QiskitRuntimeService` with a safety fallback to the Aer simulator when hardware queues are unavailable.
   - Returns results suitable for telemetry ingestion.

   Sample usage (inside a Qallow phase driver or notebook):
   ```python
   from python.quantum.qallow_ibm_bridge import run_ternary_sim

   results = run_ternary_sim([-1, 0, 1])
   print(results.counts)
   ```

4. **Operational run**:
   ```bash
   ./build/qallow --phase=11 --ticks=400
   ```
   With the bridge enabled, Phase 11 steps can invoke IBM Quantum hardware or simulators when ternary coherence checks request external validation. Monitor `data/logs/telemetry.csv` (or the relevant telemetry sink) for quantum result entries.

5. **Fallback and ethics hooks**:
   - If the hardware queue is unavailable, the bridge automatically redirects to `AerSimulator` so Qallow workloads keep running.
   - Integrate the ethics module (Phase 9) by scoring returned distributions before they influence autonomous routines. Reject runs with scores below `0.94` and trigger token rotation if anomalies appear.

## Verification Checklist
- [ ] `QiskitRuntimeService().backends()` returns a list without authentication errors.
- [ ] `python examples/ibm_quantum_bell.py` completes and prints measurement statistics.
- [ ] `./build/qallow --phase=11 --ticks=400` emits telemetry entries that reference IBM Quantum results.
- [ ] Tokens are stored outside version control (`.env`, shell profile, or Qiskit credential store).

## Troubleshooting
- **Queue delays**: Use `service.least_busy(simulator=False)` or specify `backend_name="ibmq_qasm_simulator"` during high load periods.
- **Invalid token**: Regenerate in the dashboard, update local storage, and rerun the verification steps.
- **Missing dependencies**: Reinstall with `pip install --upgrade qiskit qiskit-ibm-runtime`. Confirm that the active Python interpreter matches the environment used by Qallow.
- **Network restrictions**: Ensure outbound HTTPS to `quantum-computing.ibm.com` is permitted. If running in an isolated environment, prefetch job results from a connected host via the SDK.

## Next Steps
- Draft an RFC in `docs/rfcs/` if you plan to extend the bridge for dedicated qutrit encodings or custom transpilation passes.
- Consider scheduling token rotations and backend availability checks via the monitoring stack under `monitoring/` to catch credential or service drift early.
