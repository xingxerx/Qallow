#!/usr/bin/env python3
"""
Quick-start runner for the Qallow domain pipelines.

Each pipeline returns metrics plus an ethical compliance report. This script can be
invoked standalone or imported into orchestration notebooks.
"""




if __package__ is None or __package__ == "":
    # Allow execution via `python application_runner.py`
    current_dir = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(current_dir))
    sys.path.insert(0, str(current_dir.parent))
    from applications import (  # type: ignore
        AiAccelerationPipeline,
        ClimateModelingPipeline,
        DrugDiscoveryPipeline,
        OptimizationPipeline,
        SecureComputationPipeline,
    )
    from applications.ethics import CoherenceAuditor, ETHICAL_RESTRICTIONS  # type: ignore
else:
    from .applications import (
        AiAccelerationPipeline,
        ClimateModelingPipeline,
        DrugDiscoveryPipeline,
        OptimizationPipeline,
        SecureComputationPipeline,
    )
    from .applications.ethics import CoherenceAuditor, ETHICAL_RESTRICTIONS


def run_all(verbose: bool = True) -> Dict[str, Dict[str, object]]:
    auditor = CoherenceAuditor()

    pipelines = {
        "secure_computation": SecureComputationPipeline(auditor=auditor),
        "drug_discovery": DrugDiscoveryPipeline(auditor=auditor),
        "optimization": OptimizationPipeline(auditor=auditor),
        "ai_acceleration": AiAccelerationPipeline(auditor=auditor),
        "climate_modeling": ClimateModelingPipeline(auditor=auditor),
    }

    results: Dict[str, Dict[str, object]] = {}
    for domain, pipeline in pipelines.items():
        outcome = pipeline.execute()
        results[domain] = outcome
        if verbose:
            metrics = outcome["metrics"]
            report = outcome["report"]
            print(f"\n[{domain}] metrics: {metrics}")
            print(f"[{domain}] ethical pass: {report.passed}")

    if verbose:
        print("\nPolicy reminder:", ETHICAL_RESTRICTIONS)

    return results


if __name__ == "__main__":
    run_all(verbose=True)
