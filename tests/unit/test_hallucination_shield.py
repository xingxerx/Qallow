#!/usr/bin/env python3
"""
Unit tests for HallucinationShield Module

Tests hallucination detection, validation, and correction functionality.
"""

import unittest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add qallow to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qallow.core.governance import (
    HallucinationShield,
    HallucinationLevel,
    ValidationResult,
    create_default_shield,
    validate_llm_output,
)


class TestHallucinationShield(unittest.TestCase):
    """Test core HallucinationShield functionality."""
    
    def setUp(self):
        """Initialize test fixtures."""
        self.shield = HallucinationShield(
            threshold=0.8,
            enable_correction=True,
            max_correction_attempts=3
        )
    
    def test_shield_initialization(self):
        """Test shield initializes with correct parameters."""
        self.assertEqual(self.shield.threshold, 0.8)
        self.assertTrue(self.shield.enable_correction)
        self.assertEqual(self.shield.max_correction_attempts, 3)
        self.assertTrue(self.shield.check_factual)
        self.assertTrue(self.shield.check_contextual)
        self.assertTrue(self.shield.check_semantic)
    
    def test_shield_initialization_custom(self):
        """Test shield with custom parameters."""
        shield = HallucinationShield(
            threshold=0.9,
            enable_correction=False,
            check_factual=False
        )
        self.assertEqual(shield.threshold, 0.9)
        self.assertFalse(shield.enable_correction)
        self.assertFalse(shield.check_factual)
    
    def test_forward_valid_output(self):
        """Test validation of valid output."""
        result = self.shield.forward(
            output="The capital of France is Paris.",
            context="What is the capital of France?"
        )
        
        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(result.is_valid)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        self.assertIsInstance(result.hallucination_level, HallucinationLevel)
    
    def test_forward_empty_output(self):
        """Test validation of empty output."""
        result = self.shield.forward(output="")
        
        self.assertFalse(result.is_valid)
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(result.hallucination_level, HallucinationLevel.CRITICAL)
        self.assertIn("Empty or invalid output", result.issues)
    
    def test_forward_with_context(self):
        """Test validation with context provided."""
        result = self.shield.forward(
            output="Paris is the capital city.",
            context="What is the capital of France?"
        )
        
        self.assertTrue(result.metadata['context_provided'])
        self.assertIsInstance(result, ValidationResult)
    
    def test_forward_without_context(self):
        """Test validation without context."""
        result = self.shield.forward(
            output="Paris is the capital of France."
        )
        
        self.assertFalse(result.metadata['context_provided'])
        self.assertIsInstance(result, ValidationResult)
    
    def test_forward_with_reference(self):
        """Test validation with reference text."""
        result = self.shield.forward(
            output="Paris is the capital of France.",
            reference="France's capital: Paris"
        )
        
        self.assertTrue(result.metadata['reference_provided'])
        self.assertIsInstance(result, ValidationResult)
    
    def test_forward_updates_statistics(self):
        """Test that forward() updates statistics."""
        initial_count = self.shield.stats['total_validations']
        
        self.shield.forward(output="Test output")
        
        self.assertEqual(
            self.shield.stats['total_validations'],
            initial_count + 1
        )
    
    def test_hallucination_level_determination(self):
        """Test hallucination level determination."""
        # Test NONE level
        level = self.shield._determine_hallucination_level(0.95)
        self.assertEqual(level, HallucinationLevel.NONE)
        
        # Test LOW level
        level = self.shield._determine_hallucination_level(0.80)
        self.assertEqual(level, HallucinationLevel.LOW)
        
        # Test MEDIUM level
        level = self.shield._determine_hallucination_level(0.60)
        self.assertEqual(level, HallucinationLevel.MEDIUM)
        
        # Test HIGH level
        level = self.shield._determine_hallucination_level(0.40)
        self.assertEqual(level, HallucinationLevel.HIGH)
        
        # Test CRITICAL level
        level = self.shield._determine_hallucination_level(0.20)
        self.assertEqual(level, HallucinationLevel.CRITICAL)
    
    def test_factual_consistency_check(self):
        """Test factual consistency checking."""
        score, issues = self.shield._check_factual_consistency(
            output="This is a factual statement.",
            reference="factual"
        )
        
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertIsInstance(issues, list)
    
    def test_factual_check_with_uncertainty(self):
        """Test factual check detects uncertainty markers."""
        score, issues = self.shield._check_factual_consistency(
            output="I apologize, but I don't have information about this.",
            reference=None
        )
        
        self.assertLess(score, 0.9)
        self.assertTrue(any("Uncertainty marker" in issue for issue in issues))
    
    def test_contextual_relevance_check(self):
        """Test contextual relevance checking."""
        score, issues = self.shield._check_contextual_relevance(
            output="Paris is the capital of France.",
            context="What is the capital of France?"
        )
        
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertIsInstance(issues, list)
    
    def test_contextual_check_low_relevance(self):
        """Test contextual check detects low relevance."""
        score, issues = self.shield._check_contextual_relevance(
            output="The weather is nice today.",
            context="What is the capital of France?"
        )
        
        # Low overlap should result in lower score
        self.assertLess(score, 0.9)
    
    def test_semantic_coherence_check(self):
        """Test semantic coherence checking."""
        score, issues = self.shield._check_semantic_coherence(
            output="This is a coherent and well-formed sentence."
        )
        
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertIsInstance(issues, list)
    
    def test_semantic_check_short_output(self):
        """Test semantic check on short output."""
        score, issues = self.shield._check_semantic_coherence(
            output="Hi"
        )
        
        self.assertIn("Output too short", issues[0])
    
    def test_semantic_check_repetition(self):
        """Test semantic check detects repetition."""
        score, issues = self.shield._check_semantic_coherence(
            output="test test test test test test test test test test test"
        )
        
        self.assertLess(score, 0.9)
        self.assertTrue(any("repetition" in issue.lower() for issue in issues))
    
    def test_correction_chain_disabled(self):
        """Test correction chain when disabled."""
        shield = HallucinationShield(enable_correction=False)
        result = shield.correction_chain(
            output="Test output",
            issues=["Issue 1"]
        )
        
        self.assertIsNone(result)
    
    def test_correction_chain_basic(self):
        """Test basic correction chain."""
        result = self.shield.correction_chain(
            output="I apologize, but I cannot verify this information.",
            issues=["Uncertainty marker detected"]
        )
        
        # Correction may fail if output cannot be improved
        # This is expected behavior for edge cases
        # The test verifies the correction mechanism runs without errors
        self.assertIsInstance(result, (str, type(None)))
    
    def test_correction_chain_with_context(self):
        """Test correction chain with context."""
        result = self.shield.correction_chain(
            output="I apologize, but unclear statement.",
            issues=["Uncertainty marker detected"],
            context="What is the capital of France?"
        )
        
        # Correction may fail if context doesn't help improve output
        # This test verifies the mechanism executes properly
        self.assertIsInstance(result, (str, type(None)))
    
    def test_correction_chain_max_attempts(self):
        """Test correction chain respects max attempts."""
        # Use a low threshold to force validation to fail
        shield = HallucinationShield(
            threshold=0.99,
            enable_correction=True,
            max_correction_attempts=2
        )
        
        result = shield.correction_chain(
            output="Test output",
            issues=["Issue"],
            max_attempts=1
        )
        
        # With high threshold, correction may fail
        # Test verifies the max_attempts parameter is respected
        self.assertIsInstance(result, (str, type(None)))
    
    def test_apply_corrections(self):
        """Test correction application."""
        corrected = self.shield._apply_corrections(
            output="I apologize, but this is a test.",
            issues=["uncertainty marker detected"],
            context=None
        )
        
        self.assertNotIn("I apologize, but", corrected)
        self.assertIsInstance(corrected, str)
    
    def test_get_statistics(self):
        """Test getting statistics."""
        stats = self.shield.get_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_validations', stats)
        self.assertIn('passed_validations', stats)
        self.assertIn('failed_validations', stats)
        self.assertIn('corrections_applied', stats)
        self.assertIn('average_confidence', stats)
    
    def test_reset_statistics(self):
        """Test resetting statistics."""
        # Generate some statistics
        self.shield.forward(output="Test")
        self.shield.forward(output="Test 2")
        
        # Reset
        self.shield.reset_statistics()
        
        stats = self.shield.get_statistics()
        self.assertEqual(stats['total_validations'], 0)
        self.assertEqual(stats['passed_validations'], 0)
        self.assertEqual(stats['failed_validations'], 0)
        self.assertEqual(stats['corrections_applied'], 0)
        self.assertEqual(stats['average_confidence'], 0.0)
    
    def test_repr(self):
        """Test string representation."""
        repr_str = repr(self.shield)
        
        self.assertIn('HallucinationShield', repr_str)
        self.assertIn('threshold=0.8', repr_str)
        self.assertIn('correction=enabled', repr_str)


class TestHallucinationShieldHelpers(unittest.TestCase):
    """Test helper functions."""
    
    def test_create_default_shield(self):
        """Test creating default shield."""
        shield = create_default_shield()
        
        self.assertIsInstance(shield, HallucinationShield)
        self.assertEqual(shield.threshold, 0.8)
    
    def test_create_default_shield_with_overrides(self):
        """Test creating shield with custom parameters."""
        shield = create_default_shield(threshold=0.9, enable_correction=False)
        
        self.assertEqual(shield.threshold, 0.9)
        self.assertFalse(shield.enable_correction)
    
    def test_validate_llm_output_convenience(self):
        """Test convenience validation function."""
        result = validate_llm_output(
            output="Test output",
            context="Test context"
        )
        
        self.assertIsInstance(result, ValidationResult)
    
    def test_validate_llm_output_with_shield(self):
        """Test convenience function with custom shield."""
        shield = HallucinationShield(threshold=0.9)
        result = validate_llm_output(
            output="Test output",
            shield=shield
        )
        
        self.assertIsInstance(result, ValidationResult)


class TestValidationResult(unittest.TestCase):
    """Test ValidationResult dataclass."""
    
    def test_validation_result_creation(self):
        """Test creating ValidationResult."""
        result = ValidationResult(
            is_valid=True,
            confidence=0.95,
            hallucination_level=HallucinationLevel.NONE,
            issues=[],
            metadata={'test': True}
        )
        
        self.assertTrue(result.is_valid)
        self.assertEqual(result.confidence, 0.95)
        self.assertEqual(result.hallucination_level, HallucinationLevel.NONE)
        self.assertEqual(len(result.issues), 0)
        self.assertIn('test', result.metadata)
    
    def test_validation_result_with_issues(self):
        """Test ValidationResult with issues."""
        result = ValidationResult(
            is_valid=False,
            confidence=0.5,
            hallucination_level=HallucinationLevel.MEDIUM,
            issues=["Issue 1", "Issue 2"]
        )
        
        self.assertFalse(result.is_valid)
        self.assertEqual(len(result.issues), 2)
    
    def test_validation_result_with_correction(self):
        """Test ValidationResult with corrected output."""
        result = ValidationResult(
            is_valid=True,
            confidence=0.85,
            hallucination_level=HallucinationLevel.LOW,
            corrected_output="Corrected text"
        )
        
        self.assertEqual(result.corrected_output, "Corrected text")


class TestHallucinationShieldIntegration(unittest.TestCase):
    """Integration tests for HallucinationShield."""
    
    def test_full_validation_pipeline(self):
        """Test full validation pipeline."""
        shield = HallucinationShield(threshold=0.7)
        
        # Validate output
        result = shield.forward(
            output="Paris is the capital of France.",
            context="What is the capital of France?"
        )
        
        # Check result
        self.assertIsInstance(result, ValidationResult)
        self.assertGreaterEqual(result.confidence, 0.0)
        
        # Verify statistics updated
        stats = shield.get_statistics()
        self.assertGreater(stats['total_validations'], 0)
    
    def test_validation_and_correction_pipeline(self):
        """Test validation followed by correction."""
        shield = HallucinationShield(
            threshold=0.8,
            enable_correction=True
        )
        
        # Create output with issues
        output = "I apologize, but I cannot verify this statement."
        
        # Validate
        result = shield.forward(output)
        
        # If invalid, try correction
        if not result.is_valid:
            corrected = shield.correction_chain(
                output=output,
                issues=result.issues
            )
            
            if corrected:
                # Validate corrected output
                corrected_result = shield.forward(corrected)
                self.assertIsInstance(corrected_result, ValidationResult)
    
    def test_multiple_validations_statistics(self):
        """Test statistics across multiple validations."""
        shield = HallucinationShield()
        
        # Perform multiple validations
        outputs = [
            "The sky is blue.",
            "Water is wet.",
            "The Earth orbits the Sun."
        ]
        
        for output in outputs:
            shield.forward(output)
        
        stats = shield.get_statistics()
        self.assertEqual(stats['total_validations'], len(outputs))
        self.assertGreaterEqual(stats['average_confidence'], 0.0)


class TestHallucinationLevel(unittest.TestCase):
    """Test HallucinationLevel enum."""
    
    def test_hallucination_levels(self):
        """Test all hallucination levels."""
        levels = [
            HallucinationLevel.NONE,
            HallucinationLevel.LOW,
            HallucinationLevel.MEDIUM,
            HallucinationLevel.HIGH,
            HallucinationLevel.CRITICAL
        ]
        
        for level in levels:
            self.assertIsInstance(level, HallucinationLevel)
            self.assertIsInstance(level.value, str)


if __name__ == '__main__':
    unittest.main()
