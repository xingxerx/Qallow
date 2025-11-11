"""
Unit tests for the memory system

Tests the experience store, schemas, and ContReAct loop integration.
"""

import pytest
import asyncio
from datetime import datetime
import uuid

from qallow.memory.experience_store import ExperienceStore, ExperiencePayload
from qallow.memory.schemas import AgentRole, OutcomeStatus
from qallow.memory.react_loop import ContReActLoop
from qallow.memory.telemetry import TelemetryManager, get_telemetry_manager


class TestExperiencePayload:
    """Test ExperiencePayload dataclass"""
    
    def test_create_payload(self):
        """Test creating an experience payload"""
        payload = ExperiencePayload(
            experience_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat(),
            agent_role="Code_Verifier",
            agent_action="Tool:run_unit_tests",
            outcome_status="Success",
            context_summary="Test passed successfully",
            prompt_version="V1.0"
        )
        
        assert payload.agent_role == "Code_Verifier"
        assert payload.outcome_status == "Success"
        assert payload.trace_link is None
    
    def test_payload_to_dict(self):
        """Test converting payload to dictionary"""
        payload = ExperiencePayload(
            experience_id="test-id",
            timestamp="2025-11-11T10:00:00",
            agent_role="Optimizer",
            agent_action="optimize_params",
            outcome_status="Partial",
            context_summary="Partial optimization",
            prompt_version="V1.1",
            trace_link="vscode://trace/123"
        )
        
        payload_dict = payload.to_dict()
        assert payload_dict["experience_id"] == "test-id"
        assert payload_dict["agent_role"] == "Optimizer"
        assert payload_dict["trace_link"] == "vscode://trace/123"


class TestExperienceStore:
    """Test ExperienceStore functionality"""
    
    def test_store_initialization(self):
        """Test initializing the experience store"""
        store = ExperienceStore(use_embeddings=False)
        assert store.qdrant_url == "http://localhost:6333"
        assert store.use_embeddings is False
    
    @pytest.mark.asyncio
    async def test_generate_embedding(self):
        """Test embedding generation"""
        store = ExperienceStore(use_embeddings=True)
        
        # Test with embeddings available
        if store.embedder:
            embedding = await store.generate_embedding("test text")
            assert isinstance(embedding, list)
            assert len(embedding) == 384  # all-MiniLM-L6-v2 dimension
        else:
            # Test fallback to zero vector
            embedding = await store.generate_embedding("test text")
            assert isinstance(embedding, list)
            assert len(embedding) == 384
    
    @pytest.mark.asyncio
    async def test_write_experience_without_qdrant(self):
        """Test writing experience when Qdrant is not available"""
        store = ExperienceStore(use_embeddings=False)
        
        payload = ExperiencePayload(
            experience_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat(),
            agent_role="Code_Verifier",
            agent_action="test_action",
            outcome_status="Success",
            context_summary="Test experience",
            prompt_version="V1.0"
        )
        
        # Should return False when Qdrant is not available
        result = await store.write_experience(payload)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_retrieve_experiences_without_qdrant(self):
        """Test retrieving experiences when Qdrant is not available"""
        store = ExperienceStore(use_embeddings=False)
        
        # Should return empty list when Qdrant is not available
        results = await store.retrieve_similar_experiences("test query")
        assert results == []


class TestSchemas:
    """Test schema definitions"""
    
    def test_agent_role_enum(self):
        """Test AgentRole enumeration"""
        assert AgentRole.CODE_VERIFIER.value == "Code_Verifier"
        assert AgentRole.OPTIMIZER.value == "Optimizer"
        assert AgentRole.EVALUATOR.value == "Evaluator"
    
    def test_outcome_status_enum(self):
        """Test OutcomeStatus enumeration"""
        assert OutcomeStatus.SUCCESS.value == "Success"
        assert OutcomeStatus.FAILURE.value == "Failure"
        assert OutcomeStatus.PARTIAL.value == "Partial"


class TestTelemetry:
    """Test telemetry functionality"""
    
    def test_telemetry_manager_singleton(self):
        """Test that TelemetryManager is a singleton"""
        manager1 = get_telemetry_manager()
        manager2 = get_telemetry_manager()
        assert manager1 is manager2
    
    def test_telemetry_manager_initialization(self):
        """Test initializing telemetry manager"""
        manager = TelemetryManager()
        assert manager is not None


class TestContReActLoop:
    """Test ContReAct loop functionality"""
    
    def test_loop_initialization(self):
        """Test initializing ContReAct loop"""
        store = ExperienceStore(use_embeddings=False)
        loop = ContReActLoop(
            agent_role="Code_Verifier",
            experience_store=store,
            max_iterations=5
        )
        
        assert loop.agent_role == "Code_Verifier"
        assert loop.max_iterations == 5
        assert loop.iteration_count == 0
    
    @pytest.mark.asyncio
    async def test_retrieve_context(self):
        """Test retrieving context in ContReAct loop"""
        store = ExperienceStore(use_embeddings=False)
        loop = ContReActLoop(
            agent_role="Code_Verifier",
            experience_store=store
        )
        
        # Should return empty list when Qdrant is not available
        context = await loop.retrieve_context("test query")
        assert context == []
    
    @pytest.mark.asyncio
    async def test_store_experience_in_loop(self):
        """Test storing experience in ContReAct loop"""
        store = ExperienceStore(use_embeddings=False)
        loop = ContReActLoop(
            agent_role="Code_Verifier",
            experience_store=store
        )
        
        result = await loop.store_experience(
            problem="Test problem",
            decision={"action": "test_action", "reasoning": "test reasoning"},
            observation={"status": "Success", "success": True, "summary": "Test passed"}
        )
        
        # Should return False when Qdrant is not available
        assert result is False


class TestIntegration:
    """Integration tests"""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test full memory system workflow"""
        store = ExperienceStore(use_embeddings=False)
        loop = ContReActLoop(
            agent_role="Code_Verifier",
            experience_store=store,
            max_iterations=1
        )
        
        # Define simple async functions for testing
        async def reasoning_fn(problem, context):
            return {"action": "test", "reasoning": "test reasoning"}
        
        async def action_fn(decision):
            return {"status": "completed", "result": "test result"}
        
        async def observation_fn(action_result):
            return {"success": True, "summary": "Test completed"}
        
        # Run the loop
        result = await loop.run(
            problem="Test problem",
            reasoning_fn=reasoning_fn,
            action_fn=action_fn,
            observation_fn=observation_fn
        )
        
        assert result["problem"] == "Test problem"
        assert len(result["iterations"]) > 0
        assert result["success"] is True


# Pytest configuration
@pytest.fixture
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

