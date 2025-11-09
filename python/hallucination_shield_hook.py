#!/usr/bin/env python3
"""
HallucinationShield Integration Hook Template

This module provides a ready-to-use integration template for adding
HallucinationShield validation to existing inference pipelines.

Integration Instructions:
1. Import this module in your inference pipeline
2. Wrap your LLM generation calls with validate_and_correct()
3. Configure the shield according to your needs

Example Integration:
    # In your inference module
    from python.hallucination_shield_hook import create_validated_pipeline
    
    # Wrap your existing LLM function
    validated_llm = create_validated_pipeline(your_llm_function)
    
    # Use as normal
    result = validated_llm(prompt="Your prompt here")
"""

import sys
from pathlib import Path
from typing import Callable, Dict, Any, Optional
import logging

# Add qallow to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qallow.core.governance import (
    HallucinationShield,
    HallucinationLevel,
    ValidationResult,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidatedInferencePipeline:
    """
    Wrapper class that adds HallucinationShield validation to any inference function.
    
    This provides a drop-in replacement for existing inference functions with
    automatic hallucination detection and correction.
    
    Example:
        >>> def my_llm(prompt: str) -> str:
        ...     return "LLM response"
        >>> 
        >>> validated_llm = ValidatedInferencePipeline(my_llm)
        >>> result = validated_llm("What is AI?")
    """
    
    def __init__(
        self,
        inference_fn: Callable[[str], str],
        shield_config: Optional[Dict[str, Any]] = None,
        log_validations: bool = True
    ):
        """
        Initialize validated inference pipeline.
        
        Args:
            inference_fn: Your existing LLM inference function
            shield_config: Optional HallucinationShield configuration
            log_validations: Whether to log validation results
        """
        self.inference_fn = inference_fn
        self.log_validations = log_validations
        
        # Create shield with provided config or defaults
        shield_config = shield_config or {}
        self.shield = HallucinationShield(**shield_config)
        
        logger.info(f"ValidatedInferencePipeline initialized with threshold={self.shield.threshold}")
    
    def __call__(
        self,
        prompt: str,
        auto_correct: bool = True,
        return_validation: bool = False
    ) -> str | tuple[str, ValidationResult]:
        """
        Run inference with validation.
        
        Args:
            prompt: Input prompt
            auto_correct: Whether to automatically correct issues
            return_validation: Whether to return ValidationResult along with output
        
        Returns:
            str: Final output (corrected if needed)
            or tuple[str, ValidationResult]: Output and validation result
        """
        # Step 1: Generate output
        output = self.inference_fn(prompt)
        
        # Step 2: Validate
        result = self.shield.forward(output=output, context=prompt)
        
        # Step 3: Log if enabled
        if self.log_validations:
            self._log_validation(prompt, output, result)
        
        # Step 4: Handle correction if needed
        final_output = output
        if not result.is_valid and auto_correct and self.shield.enable_correction:
            corrected = self.shield.correction_chain(
                output=output,
                issues=result.issues,
                context=prompt
            )
            if corrected:
                final_output = corrected
                result.corrected_output = corrected
                if self.log_validations:
                    logger.info(f"✅ Correction applied successfully")
        
        # Return based on return_validation flag
        if return_validation:
            return final_output, result
        return final_output
    
    def _log_validation(
        self,
        prompt: str,
        output: str,
        result: ValidationResult
    ) -> None:
        """Log validation results."""
        status = "✅ PASS" if result.is_valid else "⚠️  FAIL"
        logger.info(
            f"{status} | Confidence: {result.confidence:.2f} | "
            f"Level: {result.hallucination_level.value} | "
            f"Prompt: {prompt[:50]}..."
        )
        if result.issues:
            logger.warning(f"Issues: {result.issues}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return self.shield.get_statistics()


def create_validated_pipeline(
    inference_fn: Callable[[str], str],
    threshold: float = 0.8,
    enable_correction: bool = True,
    **kwargs
) -> ValidatedInferencePipeline:
    """
    Factory function to create a validated inference pipeline.
    
    This is the simplest way to add HallucinationShield to your pipeline.
    
    Args:
        inference_fn: Your LLM inference function
        threshold: Validation threshold (0-1)
        enable_correction: Whether to enable auto-correction
        **kwargs: Additional HallucinationShield configuration
    
    Returns:
        ValidatedInferencePipeline instance
    
    Example:
        >>> def my_llm(prompt: str) -> str:
        ...     return llm_model.generate(prompt)
        >>> 
        >>> validated_llm = create_validated_pipeline(
        ...     my_llm,
        ...     threshold=0.85,
        ...     enable_correction=True
        ... )
        >>> 
        >>> result = validated_llm("What is quantum computing?")
    """
    shield_config = {
        'threshold': threshold,
        'enable_correction': enable_correction,
        **kwargs
    }
    
    return ValidatedInferencePipeline(
        inference_fn=inference_fn,
        shield_config=shield_config
    )


def validate_and_correct(
    output: str,
    context: Optional[str] = None,
    threshold: float = 0.8,
    enable_correction: bool = True
) -> tuple[str, ValidationResult]:
    """
    Standalone function for validating and correcting a single output.
    
    Use this for one-off validations or when you don't need a full pipeline.
    
    Args:
        output: The LLM output to validate
        context: Optional context/prompt
        threshold: Validation threshold
        enable_correction: Whether to enable correction
    
    Returns:
        Tuple of (final_output, validation_result)
    
    Example:
        >>> output = "I apologize, but I don't know."
        >>> corrected, result = validate_and_correct(
        ...     output,
        ...     context="What is 2+2?",
        ...     enable_correction=True
        ... )
        >>> print(f"Valid: {result.is_valid}")
    """
    shield = HallucinationShield(
        threshold=threshold,
        enable_correction=enable_correction
    )
    
    # Validate
    result = shield.forward(output=output, context=context)
    
    # Correct if needed
    final_output = output
    if not result.is_valid and enable_correction:
        corrected = shield.correction_chain(
            output=output,
            issues=result.issues,
            context=context
        )
        if corrected:
            final_output = corrected
            result.corrected_output = corrected
    
    return final_output, result


# Example integration with existing Qallow components
def integrate_with_qallow_agi():
    """
    Example showing integration with Qallow AGI Integration module.
    
    This demonstrates how to add HallucinationShield to the existing
    qallow_agi_integration.py module.
    """
    logger.info("=" * 70)
    logger.info("HallucinationShield Integration with Qallow AGI")
    logger.info("=" * 70)
    
    # Mock AGI inference function
    def agi_inference(prompt: str) -> str:
        """Simulated AGI inference."""
        responses = {
            "explain": "I apologize, but I cannot verify this explanation.",
            "calculate": "The answer is 42.",
            "question": "This is a well-researched answer with high confidence."
        }
        for key in responses:
            if key in prompt.lower():
                return responses[key]
        return "I don't have enough information."
    
    # Create validated pipeline
    validated_agi = create_validated_pipeline(
        agi_inference,
        threshold=0.8,
        enable_correction=True
    )
    
    # Test with various prompts
    test_prompts = [
        "explain quantum computing",
        "calculate 2+2",
        "question about AI"
    ]
    
    logger.info("\nRunning test prompts through validated AGI pipeline:\n")
    
    for prompt in test_prompts:
        logger.info(f"Prompt: {prompt}")
        output, result = validated_agi(prompt, return_validation=True)
        logger.info(f"Output: {output}")
        logger.info(f"Valid: {result.is_valid}, Confidence: {result.confidence:.2f}\n")
    
    # Show statistics
    stats = validated_agi.get_statistics()
    logger.info("\nValidation Statistics:")
    logger.info(f"  Total: {stats['total_validations']}")
    logger.info(f"  Passed: {stats['passed_validations']}")
    logger.info(f"  Failed: {stats['failed_validations']}")
    logger.info(f"  Avg Confidence: {stats['average_confidence']:.3f}")


if __name__ == "__main__":
    # Run integration example
    integrate_with_qallow_agi()
    
    print("\n" + "=" * 70)
    print("Integration template ready for use!")
    print("=" * 70)
    print("\nTo integrate with your pipeline:")
    print("1. Import: from python.hallucination_shield_hook import create_validated_pipeline")
    print("2. Wrap: validated_fn = create_validated_pipeline(your_llm_function)")
    print("3. Use: result = validated_fn(prompt)")
