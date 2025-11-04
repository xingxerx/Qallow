# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] #!/usr/bin/env python3
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] """
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] Quick-start runner for the Qallow domain pipelines.
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] Each pipeline returns metrics plus an ethical compliance report. This script can be
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] invoked standalone or imported into orchestration notebooks.
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] """
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] if __package__ is None or __package__ == "":
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]     # Allow execution via `python application_runner.py`
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]     current_dir = pathlib.Path(__file__).resolve().parent
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]     sys.path.insert(0, str(current_dir))
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]     sys.path.insert(0, str(current_dir.parent))
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]     from applications import (  # type: ignore
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]         AiAccelerationPipeline,
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]         ClimateModelingPipeline,
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]         DrugDiscoveryPipeline,
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
