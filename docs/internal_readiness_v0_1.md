# Qallow Internal Readiness – v0.1

| Module | Status | Validation |
| --- | --- | --- |
| Ethics scoring core | ✅ | `tests/smoke/test_modules.sh` drives `qallow govern --ticks=4` and reports `Total Ethics Score: 2.9991` |
| Autonomous governance loop | ✅ | Governance smoke run completes with zero violations and persisted checkpoints |
| Phase 12 elasticity | ✅ | CPU build executes `qallow run --phase=12 --ticks=8`, emitting stable coherence |
| Phase 13 harmonic | ✅ | CPU build executes `qallow run --phase=13 --ticks=8 --nodes=4`, updating harmonic energy without errors |
| Phase 13 accelerator | ⚠️ | Command scripted in CI; depends on shared-memory availability and CUDA container at runtime |
| Dependency audit | ⚠️ | `nv-nsight-cu-cli` and `sentence-transformers` model flagged as missing in `scripts/check_dependencies.sh` |

## Key Metrics

| Metric | Threshold | Observed | Source |
| --- | --- | --- | --- |
| Ethics score | ≥ 0.94 | 0.9997 (2.9991 / 3.0) | `qallow govern --ticks=4` |
| Coherence | ≥ 0.998 | 0.99986 | `qallow run --phase=12 --ticks=8` |
| Fidelity | ≥ 0.981 | 0.99920 (global overlay stability) | `qallow_stream.csv` tick 0 snapshot |

## Execution Notes

- Run `tests/smoke/test_modules.sh` to build the CPU binary and exercise the ethics, governance, and phase runners.
- Accelerator mode requires `/dev/shm` and the CUDA 13.0 container (`nvidia/cuda:13.0.0-devel-ubuntu22.04`) supplied in the CI workflow.
- Use `scripts/check_dependencies.sh` to audit local prerequisites; install `nv-nsight-cu-cli` and `sentence-transformers` for a fully green report.
