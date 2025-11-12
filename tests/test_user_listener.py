#!/usr/bin/env python3
"""
Unit tests for the Qallow User Listener System

Tests cover:
- UserListener functionality
- AutoUpdater strategies
- QallowListenerIntegration
"""

import pytest
import json
import time
import tempfile
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "python" / "listeners"))

from user_listener import (
    UserListener, UserEvent, EventType, UpdateAction, get_listener
)
from auto_updater import AutoUpdater, UpdateStrategy
from qallow_listener_integration import QallowListenerIntegration


class TestUserListener:
    """Tests for UserListener class"""
    
    def setup_method(self):
        """Setup for each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.listener = UserListener(data_dir=self.temp_dir)
    
    def teardown_method(self):
        """Cleanup after each test"""
        if self.listener.running:
            self.listener.stop()
    
    def test_listener_initialization(self):
        """Test listener initialization"""
        assert self.listener is not None
        assert not self.listener.running
        assert len(self.listener.event_history) == 0
    
    def test_listener_start_stop(self):
        """Test starting and stopping listener"""
        self.listener.start()
        assert self.listener.running
        
        self.listener.stop()
        assert not self.listener.running
    
    def test_submit_event(self):
        """Test submitting an event"""
        event = UserEvent(
            event_type=EventType.USER_FEEDBACK,
            timestamp=datetime.now().isoformat(),
            user_id="test_user",
            message="Test feedback",
            metadata={"test": True},
            priority=2
        )
        
        result = self.listener.submit_event(event)
        assert result is not None
        assert result["message"] == "Test feedback"
    
    def test_event_history(self):
        """Test event history tracking"""
        self.listener.start()
        
        for i in range(5):
            event = UserEvent(
                event_type=EventType.USER_FEEDBACK,
                timestamp=datetime.now().isoformat(),
                user_id=f"user_{i}",
                message=f"Feedback {i}",
                metadata={},
                priority=1
            )
            self.listener.submit_event(event)
        
        time.sleep(1)
        
        history = self.listener.get_event_history()
        assert len(history) >= 5
    
    def test_callback_registration(self):
        """Test callback registration"""
        callback_called = []
        
        def test_callback(event: UserEvent):
            callback_called.append(event)
        
        self.listener.register_callback(EventType.USER_FEEDBACK, test_callback)
        self.listener.start()
        
        event = UserEvent(
            event_type=EventType.USER_FEEDBACK,
            timestamp=datetime.now().isoformat(),
            user_id="test",
            message="Test",
            metadata={},
            priority=1
        )
        
        self.listener.submit_event(event)
        time.sleep(1)
        
        assert len(callback_called) > 0


class TestAutoUpdater:
    """Tests for AutoUpdater class"""
    
    def setup_method(self):
        """Setup for each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.updater = AutoUpdater(repo_root=self.temp_dir, data_dir=self.temp_dir)
    
    def test_updater_initialization(self):
        """Test updater initialization"""
        assert self.updater is not None
        assert len(self.updater.strategies) > 0
    
    def test_strategy_loading(self):
        """Test that strategies are loaded"""
        assert len(self.updater.strategies) >= 3
        
        # Check for expected strategies
        event_types = [s.event_type for s in self.updater.strategies]
        assert EventType.CONFIGURATION_CHANGE in event_types
        assert EventType.PERFORMANCE_ISSUE in event_types
        assert EventType.ERROR_REPORT in event_types
    
    def test_process_config_event(self):
        """Test processing configuration change event"""
        event = UserEvent(
            event_type=EventType.CONFIGURATION_CHANGE,
            timestamp=datetime.now().isoformat(),
            user_id="test",
            message="set max_iterations = 1000",
            metadata={},
            priority=2
        )

        action = self.updater.process_event(event)
        assert action is not None
        assert action.action_type == "config_update"
        assert "key" in action.changes
        assert action.changes["key"] == "max_iterations"
    
    def test_process_performance_event(self):
        """Test processing performance issue event"""
        event = UserEvent(
            event_type=EventType.PERFORMANCE_ISSUE,
            timestamp=datetime.now().isoformat(),
            user_id="test",
            message="The system is too slow, please optimize",
            metadata={},
            priority=3
        )
        
        action = self.updater.process_event(event)
        assert action is not None
        assert action.action_type == "parameter_tune"
    
    def test_process_error_event(self):
        """Test processing error report event"""
        event = UserEvent(
            event_type=EventType.ERROR_REPORT,
            timestamp=datetime.now().isoformat(),
            user_id="test",
            message="Error: CUDA kernel failed",
            metadata={"error_code": 1},
            priority=5
        )
        
        action = self.updater.process_event(event)
        assert action is not None
        assert action.action_type == "error_patch"
    
    def test_update_history(self):
        """Test update history tracking"""
        for i in range(3):
            event = UserEvent(
                event_type=EventType.CONFIGURATION_CHANGE,
                timestamp=datetime.now().isoformat(),
                user_id="test",
                message=f"set param_{i} = {i}",
                metadata={},
                priority=1
            )
            self.updater.process_event(event)
        
        history = self.updater.get_update_history()
        assert len(history) >= 3


class TestQallowIntegration:
    """Tests for QallowListenerIntegration class"""
    
    def setup_method(self):
        """Setup for each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.integration = QallowListenerIntegration(
            repo_root=self.temp_dir,
            data_dir=self.temp_dir
        )
    
    def teardown_method(self):
        """Cleanup after each test"""
        if self.integration.listener.running:
            self.integration.stop()
    
    def test_integration_initialization(self):
        """Test integration initialization"""
        assert self.integration is not None
        assert self.integration.listener is not None
        assert self.integration.updater is not None
    
    def test_integration_start_stop(self):
        """Test starting and stopping integration"""
        self.integration.start()
        assert self.integration.listener.running
        
        self.integration.stop()
        assert not self.integration.listener.running
    
    def test_submit_feedback(self):
        """Test submitting feedback through integration"""
        self.integration.start()
        
        result = self.integration.submit_feedback(
            message="Test feedback",
            event_type=EventType.USER_FEEDBACK,
            priority=2
        )
        
        assert result is not None
        
        self.integration.stop()
    
    def test_get_status(self):
        """Test getting integration status"""
        self.integration.start()
        
        status = self.integration.get_status()
        assert status is not None
        assert "listener_running" in status
        assert status["listener_running"] is True
        
        self.integration.stop()
    
    def test_multiple_event_types(self):
        """Test handling multiple event types"""
        self.integration.start()
        
        # Submit different types of events
        self.integration.submit_feedback(
            message="Test feedback",
            event_type=EventType.USER_FEEDBACK,
            priority=1
        )
        
        self.integration.submit_feedback(
            message="System is slow",
            event_type=EventType.PERFORMANCE_ISSUE,
            priority=3
        )
        
        self.integration.submit_feedback(
            message="Error occurred",
            event_type=EventType.ERROR_REPORT,
            priority=5
        )
        
        time.sleep(1)
        
        status = self.integration.get_status()
        assert status["event_history_size"] >= 3
        
        self.integration.stop()


class TestEventTypes:
    """Tests for EventType enum"""
    
    def test_event_type_values(self):
        """Test that all event types have values"""
        for event_type in EventType:
            assert event_type.value is not None
            assert isinstance(event_type.value, str)
    
    def test_event_type_count(self):
        """Test that expected event types exist"""
        expected_types = [
            EventType.USER_FEEDBACK,
            EventType.PERFORMANCE_ISSUE,
            EventType.ERROR_REPORT,
            EventType.FEATURE_REQUEST,
            EventType.CONFIGURATION_CHANGE,
            EventType.TELEMETRY_ANOMALY,
            EventType.MANUAL_TRIGGER,
        ]
        
        for event_type in expected_types:
            assert event_type in EventType


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

