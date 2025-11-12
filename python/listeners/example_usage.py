#!/usr/bin/env python3
"""
Example usage of the Qallow User Listener System

This demonstrates how to:
1. Create and start a listener
2. Submit user feedback
3. Handle automatic updates
4. Monitor system status
"""

import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from user_listener import UserListener, UserEvent, EventType, submit_user_feedback
from auto_updater import AutoUpdater
from qallow_listener_integration import QallowListenerIntegration, get_integration


def example_basic_listener():
    """Example 1: Basic listener usage"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Listener Usage")
    print("="*70)
    
    # Create listener
    listener = UserListener()
    listener.start()
    
    # Submit some events
    print("\nSubmitting user feedback events...")
    
    submit_user_feedback(
        message="The system is running slowly on large datasets",
        event_type=EventType.PERFORMANCE_ISSUE,
        metadata={"dataset_size": "1GB", "processing_time": "45s"},
        priority=3
    )
    
    submit_user_feedback(
        message="Please add support for GPU acceleration",
        event_type=EventType.FEATURE_REQUEST,
        metadata={"feature": "GPU_SUPPORT"},
        priority=2
    )
    
    # Wait for processing
    time.sleep(2)
    
    # Check history
    history = listener.get_event_history(limit=10)
    print(f"\nEvent history ({len(history)} events):")
    for event in history:
        print(f"  - {event['event_type']}: {event['message']}")
    
    listener.stop()


def example_auto_updater():
    """Example 2: Auto-updater with strategies"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Auto-Updater with Strategies")
    print("="*70)
    
    updater = AutoUpdater()
    
    # Create test events
    print("\nProcessing events with update strategies...")
    
    events = [
        UserEvent(
            event_type=EventType.CONFIGURATION_CHANGE,
            timestamp="2024-01-01T00:00:00",
            user_id="user1",
            message="set max_iterations = 1000",
            metadata={},
            priority=2
        ),
        UserEvent(
            event_type=EventType.PERFORMANCE_ISSUE,
            timestamp="2024-01-01T00:01:00",
            user_id="user1",
            message="The optimization is too slow, please optimize the algorithm",
            metadata={"current_time": "120s"},
            priority=4
        ),
    ]
    
    for event in events:
        action = updater.process_event(event)
        if action:
            print(f"\nGenerated action: {action.action_id}")
            print(f"  Type: {action.action_type}")
            print(f"  Target: {action.target_file}")
            print(f"  Changes: {action.changes}")


def example_integration():
    """Example 3: Full integration with Qallow"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Full Qallow Integration")
    print("="*70)
    
    # Get integration instance
    integration = get_integration()
    integration.start()
    
    print("\nSubmitting various types of feedback...")
    
    # Submit different types of feedback
    integration.submit_feedback(
        message="Error: CUDA kernel failed with code 1",
        event_type=EventType.ERROR_REPORT,
        metadata={"error_code": 1, "component": "cuda_backend"},
        priority=5,
        user_id="system"
    )
    
    integration.submit_feedback(
        message="Detected anomaly in telemetry: CPU usage spike",
        event_type=EventType.TELEMETRY_ANOMALY,
        metadata={"anomaly": "cpu_spike", "value": 95},
        priority=3,
        user_id="monitor"
    )
    
    integration.submit_feedback(
        message="set learning_rate = 0.001",
        event_type=EventType.CONFIGURATION_CHANGE,
        metadata={"param": "learning_rate", "old_value": 0.01, "new_value": 0.001},
        priority=2,
        user_id="user1"
    )
    
    # Wait for processing
    time.sleep(2)
    
    # Get status
    status = integration.get_status()
    print(f"\nIntegration Status:")
    print(f"  Listener running: {status['listener_running']}")
    print(f"  Events processed: {status['event_history_size']}")
    print(f"  Updates generated: {status['update_history_size']}")
    print(f"  Strategies loaded: {status['strategies_loaded']}")
    
    integration.stop()


def example_custom_callback():
    """Example 4: Custom callback handling"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Custom Callback Handling")
    print("="*70)
    
    listener = UserListener()
    
    # Define custom callback
    def my_callback(event: UserEvent):
        print(f"\n[CUSTOM CALLBACK] Received event: {event.event_type.value}")
        print(f"  Message: {event.message}")
        print(f"  Priority: {event.priority}")
        if event.metadata:
            print(f"  Metadata: {event.metadata}")
    
    # Register callback
    listener.register_callback(EventType.USER_FEEDBACK, my_callback)
    listener.register_callback(EventType.PERFORMANCE_ISSUE, my_callback)
    
    listener.start()
    
    print("\nSubmitting events to trigger callbacks...")
    
    submit_user_feedback(
        message="This is a test feedback",
        event_type=EventType.USER_FEEDBACK,
        priority=2
    )
    
    submit_user_feedback(
        message="System performance degraded",
        event_type=EventType.PERFORMANCE_ISSUE,
        metadata={"metric": "latency", "value": "500ms"},
        priority=3
    )
    
    # Wait for processing
    time.sleep(2)
    
    listener.stop()


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("QALLOW USER LISTENER SYSTEM - EXAMPLES")
    print("="*70)
    
    try:
        example_basic_listener()
        example_auto_updater()
        example_integration()
        example_custom_callback()
        
        print("\n" + "="*70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*70)
        print("\nCheck data/listeners/ for generated logs and data files")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

