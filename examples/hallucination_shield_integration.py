#!/usr/bin/env python3
"""
HallucinationShield Integration Example

This example demonstrates how to integrate the HallucinationShield module
into the Qallow inference pipeline. It shows:
1. Initialization of the shield
2. Validation of LLM outputs
3. Correction of detected hallucinations
4. Integration with existing pipelines
"""

import sys
from pathlib import Path
from typing import Optional

# Add qallow to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qallow.core.governance import (
    HallucinationShield,
    HallucinationLevel,
    ValidationResult,
)


class MockLLMPipeline:
    """
    Mock LLM inference pipeline for demonstration.
    
    In a real implementation, this would be your actual LLM inference engine.
    """
    
    def __init__(self):
        self.shield = HallucinationShield(
            threshold=0.8,
            enable_correction=True,
            check_factual=True,
            check_contextual=True,
            check_semantic=True
        )
    
    def generate(self, prompt: str) -> str:
        """
        Mock LLM generation.
        
        In production, this would call your actual LLM.
        """
        # Simulate LLM output
        responses = {
            "What is the capital of France?": "Paris is the capital of France.",
            "Explain quantum computing": "I apologize, but I don't have information about quantum computing.",
            "Tell me about AI": "Artificial Intelligence is a field of computer science focused on creating intelligent machines.",
        }
        
        return responses.get(prompt, "I don't know.")
    
    def generate_with_validation(
        self,
        prompt: str,
        auto_correct: bool = True
    ) -> tuple[str, ValidationResult]:
        """
        Generate LLM output with hallucination validation.
        
        This is the recommended integration pattern:
        1. Generate LLM output
        2. Validate for hallucinations
        3. Optionally correct if issues found
        4. Return final output with validation result
        
        Args:
            prompt: Input prompt
            auto_correct: Whether to automatically correct detected issues
        
        Returns:
            Tuple of (final_output, validation_result)
        """
        # Step 1: Generate initial output
        output = self.generate(prompt)
        
        # Step 2: Validate the output
        result = self.shield.forward(
            output=output,
            context=prompt
        )
        
        # Step 3: Handle validation result
        final_output = output
        
        if not result.is_valid:
            print(f"⚠️  Validation failed: {result.hallucination_level.value} level hallucination")
            print(f"   Issues: {result.issues}")
            
            if auto_correct and self.shield.enable_correction:
                print("🔧 Attempting automatic correction...")
                corrected = self.shield.correction_chain(
                    output=output,
                    issues=result.issues,
                    context=prompt
                )
                
                if corrected:
                    final_output = corrected
                    result.corrected_output = corrected
                    print("✅ Correction successful")
                else:
                    print("❌ Correction failed")
        else:
            print(f"✅ Validation passed (confidence: {result.confidence:.2f})")
        
        return final_output, result


def example_basic_validation():
    """Example 1: Basic validation of LLM output."""
    print("=" * 70)
    print("EXAMPLE 1: Basic Validation")
    print("=" * 70)
    
    shield = HallucinationShield(threshold=0.8)
    
    # Validate a good output
    result = shield.forward(
        output="Paris is the capital of France.",
        context="What is the capital of France?"
    )
    
    print(f"Output: 'Paris is the capital of France.'")
    print(f"Valid: {result.is_valid}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Hallucination Level: {result.hallucination_level.value}")
    print()


def example_with_issues():
    """Example 2: Validation detecting issues."""
    print("=" * 70)
    print("EXAMPLE 2: Validation with Issues")
    print("=" * 70)
    
    shield = HallucinationShield(threshold=0.8)
    
    # Validate output with uncertainty markers
    result = shield.forward(
        output="I apologize, but I cannot verify this information about Paris.",
        context="What is the capital of France?"
    )
    
    print(f"Output: 'I apologize, but I cannot verify this information about Paris.'")
    print(f"Valid: {result.is_valid}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Hallucination Level: {result.hallucination_level.value}")
    print(f"Issues Detected: {result.issues}")
    print()


def example_with_correction():
    """Example 3: Validation with automatic correction."""
    print("=" * 70)
    print("EXAMPLE 3: Validation with Correction")
    print("=" * 70)
    
    shield = HallucinationShield(
        threshold=0.8,
        enable_correction=True
    )
    
    output = "I apologize, but this may not be accurate information."
    
    # Validate
    result = shield.forward(output)
    
    print(f"Original Output: '{output}'")
    print(f"Valid: {result.is_valid}")
    
    if not result.is_valid:
        # Attempt correction
        corrected = shield.correction_chain(
            output=output,
            issues=result.issues
        )
        
        if corrected:
            print(f"Corrected Output: '{corrected}'")
        else:
            print("Correction failed")
    print()


def example_full_pipeline():
    """Example 4: Full inference pipeline integration."""
    print("=" * 70)
    print("EXAMPLE 4: Full Pipeline Integration")
    print("=" * 70)
    
    pipeline = MockLLMPipeline()
    
    # Test with various prompts
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing",
        "Tell me about AI",
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 70)
        
        output, result = pipeline.generate_with_validation(
            prompt=prompt,
            auto_correct=True
        )
        
        print(f"Final Output: {output}")
        print()


def example_statistics():
    """Example 5: Tracking validation statistics."""
    print("=" * 70)
    print("EXAMPLE 5: Validation Statistics")
    print("=" * 70)
    
    shield = HallucinationShield(threshold=0.8)
    
    # Perform multiple validations
    test_outputs = [
        "This is a valid statement.",
        "Another valid output.",
        "I apologize, but I cannot verify this.",
        "This is also valid.",
        "I'm not certain about this information.",
    ]
    
    for output in test_outputs:
        shield.forward(output)
    
    # Get statistics
    stats = shield.get_statistics()
    
    print(f"Total Validations: {stats['total_validations']}")
    print(f"Passed: {stats['passed_validations']}")
    print(f"Failed: {stats['failed_validations']}")
    print(f"Corrections Applied: {stats['corrections_applied']}")
    print(f"Average Confidence: {stats['average_confidence']:.3f}")
    print()


def example_custom_configuration():
    """Example 6: Custom shield configuration."""
    print("=" * 70)
    print("EXAMPLE 6: Custom Configuration")
    print("=" * 70)
    
    # Configure shield for strict validation
    strict_shield = HallucinationShield(
        threshold=0.95,  # Very high threshold
        enable_correction=True,
        max_correction_attempts=5,
        check_factual=True,
        check_contextual=True,
        check_semantic=True
    )
    
    print("Strict Shield Configuration:")
    print(f"  Threshold: {strict_shield.threshold}")
    print(f"  Correction: {'Enabled' if strict_shield.enable_correction else 'Disabled'}")
    print(f"  Max Attempts: {strict_shield.max_correction_attempts}")
    print()
    
    # Configure shield for lenient validation
    lenient_shield = HallucinationShield(
        threshold=0.5,  # Low threshold
        enable_correction=False,
        check_factual=True,
        check_contextual=False,  # Skip contextual check
        check_semantic=True
    )
    
    print("Lenient Shield Configuration:")
    print(f"  Threshold: {lenient_shield.threshold}")
    print(f"  Correction: {'Enabled' if lenient_shield.enable_correction else 'Disabled'}")
    print(f"  Contextual Check: {'Enabled' if lenient_shield.check_contextual else 'Disabled'}")
    print()


def main():
    """Run all examples."""
    print("\n")
    print("#" * 70)
    print("# HallucinationShield Integration Examples")
    print("#" * 70)
    print()
    
    example_basic_validation()
    example_with_issues()
    example_with_correction()
    example_full_pipeline()
    example_statistics()
    example_custom_configuration()
    
    print("=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
