"""
Domain-specific application pipelines built on top of the Qallow quantum engine.
Each pipeline wraps lower-level algorithms and exposes orchestrated workflows that
can be expanded with hardware-specific integrations later on.
"""

from .secure_computation import SecureComputationPipeline
from .drug_discovery import DrugDiscoveryPipeline
from .optimization import OptimizationPipeline
from .ai_acceleration import AiAccelerationPipeline
from .climate_modeling import ClimateModelingPipeline
from .ethics import CoherenceAuditor, CoherenceReport, ETHICAL_RESTRICTIONS

__all__ = [
    "SecureComputationPipeline",
    "DrugDiscoveryPipeline",
    "OptimizationPipeline",
    "AiAccelerationPipeline",
    "ClimateModelingPipeline",
    "CoherenceAuditor",
    "CoherenceReport",
    "ETHICAL_RESTRICTIONS",
]
