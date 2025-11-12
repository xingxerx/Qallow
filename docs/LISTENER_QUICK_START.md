# Qallow User Listener System - Quick Start Guide

## Overview

The Qallow User Listener System enables the codebase to listen to user input and automatically update itself based on feedback. This guide will help you get started quickly.

## Installation

The listener system is included in the Qallow repository. No additional installation is needed.

```bash
cd /home/xing/Qallow
```

## Basic Usage

### 1. Start the Listener

```python
from python.listeners import get_listener

# Get the global listener instance
listener = get_listener()

# Start listening
listener.start()

# ... do work ...

# Stop listening
listener.stop()
```

### 2. Submit User Feedback

```python
from python.listeners import submit_user_feedback, EventType

# Submit simple feedback
submit_user_feedback(
    message="The system is running slowly",
    event_type=EventType.PERFORMANCE_ISSUE,
    priority=3
)

# Submit feedback with metadata
submit_user_feedback(
    message="Error: CUDA kernel failed",
    event_type=EventType.ERROR_REPORT,
    metadata={"error_code": 1, "component": "cuda"},
    priority=5
)
```

### 3. Use the Integration

```python
from python.listeners import get_integration

# Get integration instance
integration = get_integration()
integration.start()

# Submit feedback through integration
integration.submit_feedback(
    message="set max_iterations = 1000",
    event_type=EventType.CONFIGURATION_CHANGE,
    priority=2
)

# Get system status
status = integration.get_status()
print(f"Events processed: {status['event_history_size']}")
print(f"Updates generated: {status['update_history_size']}")

integration.stop()
```

## Event Types

The system supports these event types:

| Event Type | Use Case | Example |
|-----------|----------|---------|
| USER_FEEDBACK | General feedback | "Great feature!" |
| PERFORMANCE_ISSUE | Performance problems | "System is slow" |
| ERROR_REPORT | Errors and crashes | "CUDA kernel failed" |
| FEATURE_REQUEST | Feature requests | "Add GPU support" |
| CONFIGURATION_CHANGE | Config updates | "set param = value" |
| TELEMETRY_ANOMALY | Anomalies detected | "CPU spike detected" |
| MANUAL_TRIGGER | Manual triggers | "Run optimization" |

## Priority Levels

Events have priority levels 1-5:

- **1**: Low - Informational
- **2**: Normal - Standard feedback
- **3**: Medium - Should be addressed
- **4**: High - Needs attention
- **5**: Critical - Immediate action required

Higher priority events are processed first and may trigger automatic updates.

## Configuration

Edit `config/listener_config.yaml` to customize behavior:

```yaml
listener:
  enabled: true
  data_dir: "data/listeners"
  max_queue_size: 1000

updater:
  enabled: true
  auto_update_min_priority: 3
  dry_run: false
```

## Running Examples

Run the example script to see the system in action:

```bash
cd python/listeners
python example_usage.py
```

This demonstrates:
1. Basic listener usage
2. Auto-updater with strategies
3. Full Qallow integration
4. Custom callback handling

## Running Tests

Run the test suite:

```bash
cd /home/xing/Qallow
python -m pytest tests/test_user_listener.py -v
```

## Data Storage

All listener data is stored in `data/listeners/`:

```
data/listeners/
├── listener.log              # Listener activity
├── updater.log               # Updater activity
├── integration.log           # Integration activity
├── events_YYYYMMDD.jsonl     # Daily events
├── updates_YYYYMMDD.jsonl    # Daily updates
├── telemetry_events.jsonl    # Telemetry events
├── error_log.jsonl           # Error log
└── anomalies.jsonl           # Anomalies
```

## Common Tasks

### Register a Custom Callback

```python
from python.listeners import get_listener, EventType, UserEvent

listener = get_listener()

def my_callback(event: UserEvent):
    print(f"Event: {event.message}")
    # Do something with the event

listener.register_callback(EventType.USER_FEEDBACK, my_callback)
listener.start()
```

### Get Event History

```python
from python.listeners import get_listener

listener = get_listener()
history = listener.get_event_history(limit=50)

for event in history:
    print(f"{event['timestamp']}: {event['message']}")
```

### Get Update History

```python
from python.listeners import get_integration

integration = get_integration()
updates = integration.updater.get_update_history(limit=20)

for update in updates:
    print(f"{update['action_id']}: {update['action_type']}")
```

### Submit Configuration Change

```python
from python.listeners import submit_user_feedback, EventType

# Configuration changes follow the pattern: "set key = value"
submit_user_feedback(
    message="set learning_rate = 0.001",
    event_type=EventType.CONFIGURATION_CHANGE,
    priority=2
)
```

### Report Performance Issue

```python
from python.listeners import submit_user_feedback, EventType

submit_user_feedback(
    message="The optimization algorithm is too slow",
    event_type=EventType.PERFORMANCE_ISSUE,
    metadata={"current_time": "120s", "target_time": "30s"},
    priority=4
)
```

## Integration with Qallow Components

The listener system integrates with:

- **Telemetry System**: Logs events to telemetry
- **Memory System**: Can store feedback in memory
- **Agent System**: Agents can submit feedback
- **Configuration System**: Updates configuration files
- **Error Tracking**: Records and analyzes errors

## Troubleshooting

### Events not being processed
- Check if listener is running: `listener.running`
- Check logs in `data/listeners/listener.log`
- Verify event queue is not full

### Updates not being applied
- Check `data/listeners/updater.log` for errors
- Verify target files exist and are writable
- Check update action status

### Performance issues
- Reduce event queue size if memory is limited
- Batch process events instead of real-time
- Archive old logs periodically

## Next Steps

1. Read the full documentation in `python/listeners/README.md`
2. Explore the example code in `python/listeners/example_usage.py`
3. Review the configuration in `config/listener_config.yaml`
4. Run the tests to verify everything works
5. Integrate with your Qallow components

## Support

For issues or questions:
1. Check the logs in `data/listeners/`
2. Review the README in `python/listeners/`
3. Run the tests to verify functionality
4. Check the example code for usage patterns

## License

Part of the Qallow project. See LICENSE file for details.

