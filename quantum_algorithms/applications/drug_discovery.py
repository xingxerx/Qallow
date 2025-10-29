from dataclasses import dataclass
from typing import Dict, Optional

from quantum_algorithms.algorithms.vqe_algorithm import vqe_optimization

from .ethics import CoherenceAuditor, CoherenceReport


@dataclass
class DrugDiscoveryConfig:
    """Configuration for the molecular VQE workload."""

    n_qubits: int = 2
    n_iterations: int = 6
    classical_baseline_hours: float = 120.0
    quantum_batch_factor: float = 12.0
    target_energy_hartree: float = -1.0


class DrugDiscoveryPipeline:
    """Runs a lightweight VQE pass and records lead-time metrics."""

    def __init__(
        self,
        config: Optional[DrugDiscoveryConfig] = None,
        auditor: Optional[CoherenceAuditor] = None,
    ):
        self.config = config or DrugDiscoveryConfig()
        self.auditor = auditor or CoherenceAuditor()

    def _run_vqe_probe(self) -> Dict[str, float]:
        cfg = self.config
        energy, params = vqe_optimization(
            n_qubits=cfg.n_qubits,
            n_iterations=cfg.n_iterations,
        )

        quantum_runtime = cfg.classical_baseline_hours / cfg.quantum_batch_factor
        lead_time_improvement = cfg.classical_baseline_hours / max(quantum_runtime, 1.0)

        energy_delta = abs(cfg.target_energy_hartree - energy)

        return {
            "lead_time_improvement": lead_time_improvement,
            "energy_delta": energy_delta,
            "optimized_parameters": float(len(params)),
            "vqe_energy": float(energy),
            "iterations": float(cfg.n_iterations),
        }

    def execute(self) -> Dict[str, object]:
        metrics = self._run_vqe_probe()
        report: CoherenceReport = self.auditor.enforce("drug_discovery", metrics)
        return {
            "metrics": metrics,
            "report": report,
        }
