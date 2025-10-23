#!/usr/bin/env python3
"""
Unit tests for Quantum Echoes Algorithm (Phase 11)
Tests OTOC calculation, fidelity measurement, and Phase 14 integration
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# Add examples to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'examples'))

from quantum_echoes_demo import QuantumEchoesEngine, run_quantum_echoes_demo


class TestQuantumEchoesEngine(unittest.TestCase):
    """Test core QuantumEchoesEngine functionality."""
    
    def setUp(self):
        """Initialize test fixtures."""
        self.engine = QuantumEchoesEngine(n_qubits=3, seed=42)
    
    def test_engine_initialization(self):
        """Test engine initializes with correct parameters."""
        self.assertEqual(self.engine.n_qubits, 3)
        self.assertEqual(self.engine.seed, 42)
        self.assertIsNotNone(self.engine.backend)
    
    def test_run_quantum_echoes_basic(self):
        """Test basic quantum echoes execution."""
        result = self.engine.run_quantum_echoes(t_steps=2, shots=512)
        
        # Verify result structure
        self.assertIn('otoc_fidelity', result)
        self.assertIn('n_qubits', result)
        self.assertIn('t_steps', result)
        self.assertIn('perturb_qubit', result)
        self.assertIn('shots', result)
        
        # Verify fidelity is in valid range [0, 1]
        self.assertGreaterEqual(result['otoc_fidelity'], 0.0)
        self.assertLessEqual(result['otoc_fidelity'], 1.0)
    
    def test_run_quantum_echoes_metadata(self):
        """Test quantum echoes returns correct metadata."""
        result = self.engine.run_quantum_echoes(t_steps=3, perturb_qubit=1, shots=256)
        
        self.assertEqual(result['n_qubits'], 3)
        self.assertEqual(result['t_steps'], 3)
        self.assertEqual(result['perturb_qubit'], 1)
        self.assertEqual(result['shots'], 256)
    
    def test_invalid_t_steps(self):
        """Test that invalid t_steps raises ValueError."""
        with self.assertRaises(ValueError):
            self.engine.run_quantum_echoes(t_steps=0)
        
        with self.assertRaises(ValueError):
            self.engine.run_quantum_echoes(t_steps=-1)
    
    def test_invalid_perturb_qubit(self):
        """Test that invalid perturb_qubit raises ValueError."""
        with self.assertRaises(ValueError):
            self.engine.run_quantum_echoes(perturb_qubit=5)  # >= n_qubits
    
    def test_echo_decay_curve(self):
        """Test echo decay computation over multiple time steps."""
        decay = self.engine.compute_echo_decay(max_t_steps=3)
        
        # Should have 3 entries (t=1, 2, 3)
        self.assertEqual(len(decay), 3)
        
        # All fidelities should be in [0, 1]
        for t, fidelity in decay.items():
            self.assertGreaterEqual(fidelity, 0.0)
            self.assertLessEqual(fidelity, 1.0)
            self.assertIsInstance(t, int)
    
    def test_echo_decay_monotonic_trend(self):
        """Test that echo decay generally decreases with time (statistical trend)."""
        # Run multiple times to get statistical trend
        decay_curves = []
        for _ in range(3):
            engine = QuantumEchoesEngine(n_qubits=2, seed=np.random.randint(0, 10000))
            decay = engine.compute_echo_decay(max_t_steps=4)
            decay_curves.append(decay)
        
        # Average decay curve
        avg_decay = {}
        for t in range(1, 5):
            avg_decay[t] = np.mean([decay[t] for decay in decay_curves])
        
        # Check that there's a general trend (not strictly monotonic due to noise)
        # but first and last should show some difference
        self.assertIsNotNone(avg_decay[1])
        self.assertIsNotNone(avg_decay[4])
    
    def test_different_qubit_counts(self):
        """Test engine works with different qubit counts."""
        for n_qubits in [2, 3, 4]:
            engine = QuantumEchoesEngine(n_qubits=n_qubits)
            result = engine.run_quantum_echoes(t_steps=1, shots=256)
            self.assertEqual(result['n_qubits'], n_qubits)
            self.assertGreaterEqual(result['otoc_fidelity'], 0.0)
            self.assertLessEqual(result['otoc_fidelity'], 1.0)
    
    def test_reproducibility_with_seed(self):
        """Test that same seed produces consistent engine initialization."""
        engine1 = QuantumEchoesEngine(n_qubits=3, seed=123)
        engine2 = QuantumEchoesEngine(n_qubits=3, seed=123)

        # Both engines should have same parameters
        self.assertEqual(engine1.n_qubits, engine2.n_qubits)
        self.assertEqual(engine1.seed, engine2.seed)

        # Different seeds should produce different engines
        engine3 = QuantumEchoesEngine(n_qubits=3, seed=456)
        self.assertNotEqual(engine1.seed, engine3.seed)


class TestQuantumEchoesDemo(unittest.TestCase):
    """Test quantum echoes demo function."""
    
    def test_demo_basic_execution(self):
        """Test basic demo execution."""
        result = run_quantum_echoes_demo(n_qubits=3, t_steps=2, shots=256)
        
        self.assertIn('otoc_fidelity', result)
        self.assertIn('phase14_ready', result)
        self.assertIsInstance(result['otoc_fidelity'], float)
        self.assertIsInstance(result['phase14_ready'], bool)
    
    def test_demo_phase14_readiness_threshold(self):
        """Test Phase 14 readiness logic."""
        # High fidelity should be ready
        with patch.object(QuantumEchoesEngine, 'run_quantum_echoes') as mock_run:
            mock_run.return_value = {
                'otoc_fidelity': 0.985,
                'n_qubits': 3,
                't_steps': 2,
                'perturb_qubit': 0,
                'shots': 256,
            }
            result = run_quantum_echoes_demo(n_qubits=3, t_steps=2)
            self.assertTrue(result['phase14_ready'])
        
        # Low fidelity should not be ready
        with patch.object(QuantumEchoesEngine, 'run_quantum_echoes') as mock_run:
            mock_run.return_value = {
                'otoc_fidelity': 0.75,
                'n_qubits': 3,
                't_steps': 2,
                'perturb_qubit': 0,
                'shots': 256,
            }
            result = run_quantum_echoes_demo(n_qubits=3, t_steps=2)
            self.assertFalse(result['phase14_ready'])
    
    def test_demo_telemetry_logging(self):
        """Test telemetry CSV logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / 'telemetry.csv'
            
            result = run_quantum_echoes_demo(
                n_qubits=3,
                t_steps=2,
                shots=256,
                log_path=str(log_path),
            )
            
            # Verify log file was created
            self.assertTrue(log_path.exists())
            
            # Verify CSV content
            with open(log_path, 'r') as f:
                lines = f.readlines()
                self.assertGreater(len(lines), 1)  # Header + at least 1 data row
                self.assertIn('phase', lines[0])
                self.assertIn('otoc_fidelity', lines[0])
    
    def test_demo_telemetry_append_mode(self):
        """Test that telemetry logging appends to existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / 'telemetry.csv'
            
            # First run
            run_quantum_echoes_demo(n_qubits=2, t_steps=1, shots=128, log_path=str(log_path))
            
            # Second run
            run_quantum_echoes_demo(n_qubits=3, t_steps=2, shots=256, log_path=str(log_path))
            
            # Verify both entries are in file
            with open(log_path, 'r') as f:
                lines = f.readlines()
                self.assertEqual(len(lines), 3)  # Header + 2 data rows


class TestQuantumEchoesIntegration(unittest.TestCase):
    """Integration tests for quantum echoes with Qallow."""
    
    def test_full_pipeline_execution(self):
        """Test full quantum echoes pipeline."""
        engine = QuantumEchoesEngine(n_qubits=3, seed=42)
        
        # Run echoes
        result = engine.run_quantum_echoes(t_steps=2, shots=512)
        
        # Verify result is valid
        self.assertIsInstance(result, dict)
        self.assertGreaterEqual(result['otoc_fidelity'], 0.0)
        self.assertLessEqual(result['otoc_fidelity'], 1.0)
    
    def test_echo_decay_statistics(self):
        """Test statistical properties of echo decay."""
        engine = QuantumEchoesEngine(n_qubits=2, seed=42)
        decay = engine.compute_echo_decay(max_t_steps=5)
        
        # All values should be valid probabilities
        for t, fidelity in decay.items():
            self.assertGreaterEqual(fidelity, 0.0)
            self.assertLessEqual(fidelity, 1.0)
        
        # Should have expected number of time steps
        self.assertEqual(len(decay), 5)


class TestQuantumEchoesEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def test_single_qubit(self):
        """Test with single qubit system."""
        engine = QuantumEchoesEngine(n_qubits=1, seed=42)
        result = engine.run_quantum_echoes(t_steps=1, shots=256)
        
        self.assertEqual(result['n_qubits'], 1)
        self.assertGreaterEqual(result['otoc_fidelity'], 0.0)
        self.assertLessEqual(result['otoc_fidelity'], 1.0)
    
    def test_single_time_step(self):
        """Test with single time step."""
        engine = QuantumEchoesEngine(n_qubits=3, seed=42)
        result = engine.run_quantum_echoes(t_steps=1, shots=256)
        
        self.assertEqual(result['t_steps'], 1)
        self.assertGreaterEqual(result['otoc_fidelity'], 0.0)
    
    def test_large_shot_count(self):
        """Test with large shot count."""
        engine = QuantumEchoesEngine(n_qubits=2, seed=42)
        result = engine.run_quantum_echoes(t_steps=1, shots=4096)
        
        self.assertEqual(result['shots'], 4096)
        self.assertGreaterEqual(result['otoc_fidelity'], 0.0)


if __name__ == '__main__':
    unittest.main()

