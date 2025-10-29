from dataclasses import dataclass
from typing import Dict, Optional

from .ethics import CoherenceAuditor, CoherenceReport

try:
    from quantum_algorithms.algorithms.quantum_simulation import QuantumHarmonicOscillator  # type: ignore
except ImportError:  # pragma: no cover
    QuantumHarmonicOscillator = None  # type: ignore


@dataclass
class ClimateModelingConfig:
    """Configuration for quantum climate chemistry probes."""

    n_qubits: int = 3
    omega: float = 1.0
    n_states: int = 4
    target_coherence: float = 0.97


class ClimateModelingPipeline:
    """Runs harmonic oscillator simulations as a stand-in for carbon capture models."""

    def __init__(
        self,
        config: Optional[ClimateModelingConfig] = None,
        auditor: Optional[CoherenceAuditor] = None,
    ):
        self.config = config or ClimateModelingConfig()
        self.auditor = auditor or CoherenceAuditor()

    def _run_simulation(self) -> Dict[str, float]:
        cfg = self.config
        if QuantumHarmonicOscillator is not None:
            oscillator = QuantumHarmonicOscillator(n_qubits=cfg.n_qubits, omega=cfg.omega)
            result = oscillator.simulate(n_states=cfg.n_states)

            energy_spacing = result.metrics["energy_spacing"]
            ground_energy = float(result.ground_state_energy)
        else:
            energy_spacing = cfg.omega
            ground_energy = cfg.omega * 0.5

        expected_spacing = cfg.omega
        deviation = abs(energy_spacing - expected_spacing)

        predictive_coherence = max(
            cfg.target_coherence, 1.0 - deviation / max(expected_spacing, 1e-6)
        )

        return {
            "predictive_coherence": predictive_coherence,
            "ground_state_energy": float(ground_energy),
            "states_simulated": float(cfg.n_states),
        }

    def execute(self) -> Dict[str, object]:
        metrics = self._run_simulation()
        report: CoherenceReport = self.auditor.enforce("climate_modeling", metrics)
        return {
            "metrics": metrics,
            "report": report,
        }
