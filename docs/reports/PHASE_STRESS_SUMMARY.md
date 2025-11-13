# Phase 12 & 14 Stress Summary

Date: 2025-11-13

## Phase 14 (Coherence-lattice)
- 10k run: target fidelity reached 0.981000 [OK]
- 50k run: fidelity smoothly converged to 0.981000 [OK]; alpha≈1.935e-05
- Logs:
  - data/logs/phase14_stress_10k.log
  - data/logs/phase14_stress_50k.log
- Plots:
  - data/plots/phase14_fidelity_10k.png
  - data/plots/phase14_fidelity_50k.png
  - data/plots/phase14_fidelity_overlay.png

## Phase 12 (Elasticity)
- 10k run: Coherence≈1.000000, Decoherence≈~2e-6
- 50k run: Coherence≈1.000000, Decoherence≈~2e-6 (stable)
- CSV Artifacts:
  - data/logs/phase12_10k.csv
  - data/logs/phase12_50k.csv
- Plots:
  - data/plots/phase12_coherence_10k.png
  - data/plots/phase12_coherence_overlay.png

## Notes
- Phase 12 CLI expects space-delimited ticks: `--ticks 10000` (not `--ticks=10000`).
- Default Phase 12 CSV path is `data/logs/phase12.csv`; backed up per run to avoid overwrite.

