#!/usr/bin/env python3
"""
Integration tests for Ollama Agent
Tests Phase 13 ethics validation and Phase 14 QAOA optimization
"""

import sys
import json
import pytest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from python.agents.qallow_agent_ollama import (
    OllamaAgent,
    OllamaConfig,
    AgentTask
)


class TestOllamaAgent:
    """Test suite for Ollama agent"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return OllamaConfig(
            model="llama2:70b",
            num_gpu=1,  # Use 1 GPU for testing
            qaoa_nodes=16,  # Small for testing
            qaoa_target_fidelity=0.95,
            ethics_enabled=False,  # Disable for unit tests
            output_file=Path("data/quantum/test_agent_output.jsonl"),
            gain_json=Path("data/quantum/test_gain.json")
        )
    
    def test_config_creation(self, config):
        """Test configuration creation"""
        assert config.model == "llama2:70b"
        assert config.num_gpu == 1
        assert config.qaoa_nodes == 16
        assert config.qaoa_target_fidelity == 0.95
    
    def test_agent_initialization(self, config):
        """Test agent initialization"""
        try:
            agent = OllamaAgent(config)
            assert agent.config.model == config.model
            assert agent.session_id.startswith("agent_")
        except RuntimeError as e:
            # Ollama may not be running in CI
            pytest.skip(f"Ollama not available: {e}")
    
    def test_json_extraction(self, config):
        """Test JSON extraction from LLM response"""
        agent = OllamaAgent(config)
        
        # Test with clean JSON
        text = '{"p": 3, "gamma": 0.5, "beta": 0.3, "alpha_eff": 0.005, "reasoning": "test"}'
        result = agent._extract_json(text)
        assert result["p"] == 3
        assert result["gamma"] == 0.5
        
        # Test with surrounding text
        text = 'Here is the result: {"p": 2, "gamma": 0.4, "beta": 0.2, "alpha_eff": 0.003, "reasoning": "test"} Done.'
        result = agent._extract_json(text)
        assert result["p"] == 2
    
    def test_param_validation(self, config):
        """Test QAOA parameter validation"""
        agent = OllamaAgent(config)
        
        # Test clamping
        params = {
            "p": 100,  # Too high
            "gamma": 1.5,  # Too high
            "beta": -0.1,  # Too low
            "alpha_eff": 0.1,  # Too high
            "reasoning": "test"
        }
        
        validated = agent._validate_qaoa_params(params)
        
        assert 1 <= validated["p"] <= 10
        assert 0.0 <= validated["gamma"] <= 1.0
        assert 0.0 <= validated["beta"] <= 1.0
        assert 0.001 <= validated["alpha_eff"] <= 0.01
    
    def test_gain_export(self, config, tmp_path):
        """Test gain export to JSON"""
        config.gain_json = tmp_path / "test_gain.json"
        agent = OllamaAgent(config)
        
        agent._export_gain(0.005)
        
        assert config.gain_json.exists()
        
        with open(config.gain_json) as f:
            data = json.load(f)
        
        assert "alpha_eff" in data
        assert data["alpha_eff"] == 0.005
        assert "timestamp" in data
        assert "session_id" in data
    
    def test_status(self, config):
        """Test agent status"""
        try:
            agent = OllamaAgent(config)
            status = agent.get_status()
            
            assert "session_id" in status
            assert "model" in status
            assert "tasks_completed" in status
            assert status["model"] == config.model
        except RuntimeError:
            pytest.skip("Ollama not available")
    
    @pytest.mark.integration
    def test_qaoa_optimization(self, config):
        """Integration test: Full QAOA optimization"""
        try:
            agent = OllamaAgent(config)
            result = agent.optimize_qaoa(nodes=16, target_fidelity=0.95)
            
            # Verify result structure
            assert "p" in result
            assert "gamma" in result
            assert "beta" in result
            assert "alpha_eff" in result
            assert "reasoning" in result
            
            # Verify ranges
            assert 1 <= result["p"] <= 10
            assert 0.0 <= result["gamma"] <= 1.0
            assert 0.0 <= result["beta"] <= 1.0
            assert 0.001 <= result["alpha_eff"] <= 0.01
            
            # Verify gain file was created
            assert config.gain_json.exists()
            
        except RuntimeError as e:
            pytest.skip(f"Ollama not available: {e}")


class TestPhase13Integration:
    """Test Phase 13 ethics integration"""
    
    @pytest.fixture
    def config_with_ethics(self):
        """Config with ethics enabled"""
        return OllamaConfig(
            model="llama2:70b",
            ethics_enabled=True,
            ethics_threshold=0.85
        )
    
    @pytest.mark.integration
    def test_ethics_gate(self, config_with_ethics):
        """Test Phase 13 ethics gate"""
        try:
            agent = OllamaAgent(config_with_ethics)
            
            # Test benign prompt
            result = agent._run_phase13_ethics("Optimize quantum parameters")
            assert "pass" in result
            assert "score" in result
            
        except RuntimeError:
            pytest.skip("Qallow binary or Ollama not available")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])

