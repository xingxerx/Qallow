# Phase 13 Voice-Mode Tutorial Script

This script walks a presenter through a four-particle entanglement demo using the
new QuTiP/Qiskit bridge.

1. **Intro (0:00‑0:30)**  
   - "Welcome to Qallow Phase 13. Today we'll grow a four-qubit GHZ state and show how the VM keeps coherence above 0.999."  
   - Highlight the free-tier access: "All of this runs with open-source tools—QuTiP for modeling, Qiskit or Cirq for validation."

2. **Baseline Run (0:30‑1:30)**  
   - Execute `./scripts/baseline_benchmark.sh`.  
   - Narrate the output: throughput, coherence, decoherence.  
   - Mention the new metrics collector exposed via `qallow_get_last_run_metrics()`.

3. **Generate Entanglement (1:30‑2:30)**  
   - Run `qallow run entangle --state=ghz --validate`.  
   - Explain the probability table (peaks at `|0000⟩` and `|1111⟩`).  
   - Note the backend in use—"If Qiskit is installed you'll see `backend=qiskit`; otherwise we fall back to Cirq."

4. **Seed the VM (2:30‑3:30)**  
   - Export `QALLOW_ENTANGLEMENT_BOOTSTRAP=ghz` and rerun `qallow run vm --dashboard=10`.  
   - Point out how the pockets now start from the entanglement seed.

5. **Noise Suppression (3:30‑4:30)**  
   - Set `QALLOW_LINDBLAD_GAMMA=0.12`.  
   - Re-run the VM and read the reduced decoherence from the dashboard.

6. **Web API Preview (4:30‑5:00)**  
   - Start `python -m python.quantum.web_api`.  
   - Curl `http://127.0.0.1:8713/entangle?state=w&validate=1` and call out the JSON response.

7. **Closing (5:00‑5:30)**  
   - Summarize the metrics, remind viewers how to access everything for free, and prompt them to try the tutorials on iOS/Android once published.

> Tip: Record separate voice tracks for narration and CLI output. Keep the background music subtle to preserve clarity in the ethics/compliance sections.
