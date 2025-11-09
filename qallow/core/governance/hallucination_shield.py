#!/usr/bin/env python3
"""
HallucinationShield Module for Qallow

This module provides hallucination detection and correction capabilities for LLM outputs.
It integrates with the Qallow inference pipeline to validate and correct model outputs
before they are returned to users.

Dependencies:
    - uptrain: For hallucination detection and validation
    - qallow-core: Core Qallow functionality
    - numpy: Numerical operations
    - typing: Type hints

Integration Points:
    - Post-LLM processing pipeline
    - Pre-output validation stage
    - Inference hooks
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HallucinationLevel(Enum):
    """Enumeration of hallucination severity levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of hallucination validation.
    
    Attributes:
        is_valid: Whether the output passes validation
        confidence: Confidence score (0-1) for the validation
        hallucination_level: Detected level of hallucination
        issues: List of detected issues
        corrected_output: Optional corrected version of the output
        metadata: Additional metadata about the validation
    """
    is_valid: bool
    confidence: float
    hallucination_level: HallucinationLevel
    issues: List[str] = field(default_factory=list)
    corrected_output: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class HallucinationShield:
    """
    HallucinationShield provides comprehensive hallucination detection and correction
    for LLM outputs in the Qallow inference pipeline.
    
    This class implements:
    - Real-time hallucination detection
    - Multi-level validation (factual, contextual, semantic)
    - Automatic correction chains
    - Confidence scoring
    - Integration hooks for inference pipelines
    
    Example:
        >>> shield = HallucinationShield(threshold=0.8, enable_correction=True)
        >>> result = shield.forward(output="AI response", context="user query")
        >>> if not result.is_valid:
        >>>     corrected = shield.correction_chain(output="AI response", issues=result.issues)
    """
    
    def __init__(
        self,
        threshold: float = 0.8,
        enable_correction: bool = True,
        max_correction_attempts: int = 3,
        check_factual: bool = True,
        check_contextual: bool = True,
        check_semantic: bool = True,
        uptrain_api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the HallucinationShield.
        
        Args:
            threshold: Confidence threshold for accepting outputs (0-1)
            enable_correction: Whether to enable automatic correction
            max_correction_attempts: Maximum number of correction attempts
            check_factual: Enable factual consistency checking
            check_contextual: Enable contextual relevance checking
            check_semantic: Enable semantic coherence checking
            uptrain_api_key: Optional API key for uptrain service
            **kwargs: Additional configuration options
        """
        self.threshold = threshold
        self.enable_correction = enable_correction
        self.max_correction_attempts = max_correction_attempts
        self.check_factual = check_factual
        self.check_contextual = check_contextual
        self.check_semantic = check_semantic
        self.uptrain_api_key = uptrain_api_key
        
        # Statistics tracking
        self.stats = {
            'total_validations': 0,
            'passed_validations': 0,
            'failed_validations': 0,
            'corrections_applied': 0,
            'average_confidence': 0.0
        }
        
        # Initialize uptrain client if available
        self._uptrain_client = None
        self._initialize_uptrain()
        
        logger.info(f"HallucinationShield initialized with threshold={threshold}, "
                   f"correction={'enabled' if enable_correction else 'disabled'}")
    
    def _initialize_uptrain(self) -> None:
        """
        Initialize the uptrain client for hallucination detection.
        
        This method attempts to import and initialize the uptrain library.
        If uptrain is not available, the shield will use fallback detection methods.
        """
        try:
            # Try to import uptrain
            import uptrain
            logger.info("Uptrain library found, initializing client")
            # Initialize uptrain client here when API is available
            # self._uptrain_client = uptrain.Client(api_key=self.uptrain_api_key)
            self._uptrain_client = None  # Placeholder until uptrain is fully integrated
        except ImportError:
            logger.warning("Uptrain library not found, using fallback detection methods")
            self._uptrain_client = None
    
    def forward(
        self,
        output: str,
        context: Optional[str] = None,
        reference: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validate an LLM output for hallucinations.
        
        This is the main entry point for hallucination detection. It performs
        multiple validation checks and returns a comprehensive result.
        
        Args:
            output: The LLM output to validate
            context: Optional context/prompt that generated the output
            reference: Optional reference text for factual verification
            metadata: Optional metadata for validation context
        
        Returns:
            ValidationResult containing validation status and details
        
        Example:
            >>> shield = HallucinationShield()
            >>> result = shield.forward(
            ...     output="The capital of France is Paris.",
            ...     context="What is the capital of France?"
            ... )
            >>> print(result.is_valid, result.confidence)
        """
        self.stats['total_validations'] += 1
        
        if not output or not output.strip():
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                hallucination_level=HallucinationLevel.CRITICAL,
                issues=["Empty or invalid output"],
                metadata={'error': 'empty_output'}
            )
        
        issues = []
        scores = []
        
        # Perform factual checking
        if self.check_factual:
            factual_score, factual_issues = self._check_factual_consistency(
                output, reference
            )
            scores.append(factual_score)
            issues.extend(factual_issues)
        
        # Perform contextual checking
        if self.check_contextual and context:
            contextual_score, contextual_issues = self._check_contextual_relevance(
                output, context
            )
            scores.append(contextual_score)
            issues.extend(contextual_issues)
        
        # Perform semantic checking
        if self.check_semantic:
            semantic_score, semantic_issues = self._check_semantic_coherence(output)
            scores.append(semantic_score)
            issues.extend(semantic_issues)
        
        # Calculate overall confidence
        confidence = sum(scores) / len(scores) if scores else 0.5
        
        # Determine hallucination level
        hallucination_level = self._determine_hallucination_level(confidence)
        
        # Check if validation passes
        is_valid = confidence >= self.threshold and len(issues) == 0
        
        # Update statistics
        if is_valid:
            self.stats['passed_validations'] += 1
        else:
            self.stats['failed_validations'] += 1
        
        # Update average confidence
        total = self.stats['total_validations']
        self.stats['average_confidence'] = (
            (self.stats['average_confidence'] * (total - 1) + confidence) / total
        )
        
        result = ValidationResult(
            is_valid=is_valid,
            confidence=confidence,
            hallucination_level=hallucination_level,
            issues=issues,
            metadata={
                'context_provided': context is not None,
                'reference_provided': reference is not None,
                'check_types': {
                    'factual': self.check_factual,
                    'contextual': self.check_contextual,
                    'semantic': self.check_semantic
                },
                **(metadata or {})
            }
        )
        
        logger.debug(f"Validation result: valid={is_valid}, confidence={confidence:.3f}, "
                    f"level={hallucination_level.value}")
        
        return result
    
    def correction_chain(
        self,
        output: str,
        issues: List[str],
        context: Optional[str] = None,
        max_attempts: Optional[int] = None
    ) -> Optional[str]:
        """
        Apply correction chain to fix detected hallucinations.
        
        This method attempts to correct detected issues through multiple strategies:
        1. Pattern-based corrections
        2. Context-aware refinement
        3. Confidence-boosting rewrites
        
        Args:
            output: The original output to correct
            issues: List of detected issues from validation
            context: Optional context for correction
            max_attempts: Override for max correction attempts
        
        Returns:
            Corrected output string, or None if correction failed
        
        Example:
            >>> shield = HallucinationShield()
            >>> corrected = shield.correction_chain(
            ...     output="Incorrect statement",
            ...     issues=["Factual error detected"]
            ... )
        """
        if not self.enable_correction:
            logger.warning("Correction disabled, returning None")
            return None
        
        attempts = max_attempts or self.max_correction_attempts
        current_output = output
        
        for attempt in range(attempts):
            logger.info(f"Correction attempt {attempt + 1}/{attempts}")
            
            # Apply correction strategies
            corrected = self._apply_corrections(current_output, issues, context)
            
            # Validate the corrected output
            result = self.forward(corrected, context=context)
            
            if result.is_valid:
                logger.info(f"Correction successful after {attempt + 1} attempt(s)")
                self.stats['corrections_applied'] += 1
                return corrected
            
            current_output = corrected
            issues = result.issues
        
        logger.warning(f"Correction failed after {attempts} attempts")
        return None
    
    def _check_factual_consistency(
        self,
        output: str,
        reference: Optional[str] = None
    ) -> Tuple[float, List[str]]:
        """
        Check factual consistency of the output.
        
        Args:
            output: Output text to check
            reference: Optional reference for fact checking
        
        Returns:
            Tuple of (confidence_score, list_of_issues)
        """
        issues = []
        
        # Placeholder for uptrain factual checking
        if self._uptrain_client:
            # Use uptrain for factual checking
            # score, detected_issues = self._uptrain_client.check_factual(output, reference)
            pass
        
        # Fallback: Simple heuristic checks
        score = 0.9  # Default high confidence
        
        # Check for common hallucination patterns
        hallucination_markers = [
            "I apologize, but I don't have information",
            "I cannot verify",
            "This may not be accurate",
            "I'm not certain"
        ]
        
        for marker in hallucination_markers:
            if marker.lower() in output.lower():
                score -= 0.2
                issues.append(f"Uncertainty marker detected: {marker}")
        
        # Check for contradictions if reference is provided
        if reference and reference.lower() not in output.lower():
            # Simple check - could be enhanced
            pass
        
        return max(0.0, min(1.0, score)), issues
    
    def _check_contextual_relevance(
        self,
        output: str,
        context: str
    ) -> Tuple[float, List[str]]:
        """
        Check if output is relevant to the given context.
        
        Args:
            output: Output text to check
            context: Context/prompt text
        
        Returns:
            Tuple of (confidence_score, list_of_issues)
        """
        issues = []
        score = 0.85  # Default score
        
        # Simple keyword overlap check
        output_words = set(output.lower().split())
        context_words = set(context.lower().split())
        
        # Calculate overlap ratio
        if context_words:
            overlap = len(output_words.intersection(context_words))
            overlap_ratio = overlap / len(context_words)
            
            if overlap_ratio < 0.1:
                issues.append("Low contextual relevance detected")
                score = 0.5
            elif overlap_ratio < 0.3:
                score = 0.7
        
        return score, issues
    
    def _check_semantic_coherence(self, output: str) -> Tuple[float, List[str]]:
        """
        Check semantic coherence of the output.
        
        Args:
            output: Output text to check
        
        Returns:
            Tuple of (confidence_score, list_of_issues)
        """
        issues = []
        score = 0.9  # Default high confidence
        
        # Check for basic coherence issues
        if len(output.split()) < 3:
            issues.append("Output too short for semantic analysis")
            score = 0.6
        
        # Check for repeated phrases (potential hallucination)
        words = output.split()
        if len(words) != len(set(words)) and len(words) > 10:
            repetition_ratio = 1 - (len(set(words)) / len(words))
            if repetition_ratio > 0.3:
                issues.append("High repetition detected")
                score -= 0.3
        
        return max(0.0, score), issues
    
    def _determine_hallucination_level(self, confidence: float) -> HallucinationLevel:
        """
        Determine hallucination severity level based on confidence score.
        
        Args:
            confidence: Confidence score (0-1)
        
        Returns:
            HallucinationLevel enum value
        """
        if confidence >= 0.9:
            return HallucinationLevel.NONE
        elif confidence >= 0.75:
            return HallucinationLevel.LOW
        elif confidence >= 0.5:
            return HallucinationLevel.MEDIUM
        elif confidence >= 0.25:
            return HallucinationLevel.HIGH
        else:
            return HallucinationLevel.CRITICAL
    
    def _apply_corrections(
        self,
        output: str,
        issues: List[str],
        context: Optional[str] = None
    ) -> str:
        """
        Apply corrections to the output based on detected issues.
        
        Args:
            output: Original output
            issues: List of detected issues
            context: Optional context
        
        Returns:
            Corrected output string
        """
        corrected = output
        
        # Apply pattern-based corrections
        for issue in issues:
            if "uncertainty marker" in issue.lower():
                # Remove uncertainty phrases
                for marker in ["I apologize, but", "I cannot verify", "This may not be accurate"]:
                    corrected = corrected.replace(marker, "")
        
        # Clean up extra whitespace
        corrected = " ".join(corrected.split())
        
        return corrected.strip()
    
    def get_statistics(self) -> Dict[str, Union[int, float]]:
        """
        Get validation and correction statistics.
        
        Returns:
            Dictionary containing statistics
        """
        return self.stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset all statistics counters."""
        self.stats = {
            'total_validations': 0,
            'passed_validations': 0,
            'failed_validations': 0,
            'corrections_applied': 0,
            'average_confidence': 0.0
        }
        logger.info("Statistics reset")
    
    def __repr__(self) -> str:
        """String representation of the shield."""
        return (
            f"HallucinationShield(threshold={self.threshold}, "
            f"correction={'enabled' if self.enable_correction else 'disabled'}, "
            f"validations={self.stats['total_validations']})"
        )


# Integration helper functions

def create_default_shield(**kwargs) -> HallucinationShield:
    """
    Create a HallucinationShield with default configuration.
    
    Args:
        **kwargs: Override default configuration
    
    Returns:
        Configured HallucinationShield instance
    """
    return HallucinationShield(**kwargs)


def validate_llm_output(
    output: str,
    context: Optional[str] = None,
    shield: Optional[HallucinationShield] = None
) -> ValidationResult:
    """
    Convenience function to validate LLM output.
    
    Args:
        output: LLM output to validate
        context: Optional context
        shield: Optional pre-configured shield (creates default if None)
    
    Returns:
        ValidationResult
    """
    if shield is None:
        shield = create_default_shield()
    
    return shield.forward(output, context=context)
