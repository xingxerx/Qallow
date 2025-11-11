# Qallow User Listener System

A comprehensive system for listening to user input and automatically updating the Qallow codebase based on feedback.

## Overview

The User Listener System enables Qallow to:
- **Listen** to user interactions and feedback
- **Analyze** patterns and improvement opportunities
- **Update** the codebase automatically based on feedback
- **Track** all changes with audit trails

## Architecture

### Components

1. **UserListener** (`user_listener.py`)
   - Core listener that monitors user events
   - Manages event queue and processing
   - Maintains event history
   - Supports custom callbacks

2. **AutoUpdater** (`auto_updater.py`)
   - Processes user events and generates update actions
   - Applies configuration changes
   - Tunes parameters based on feedback
   - Maintains update audit trail

3. **QallowListenerIntegration** (`qallow_listener_integration.py`)
   - Integrates listener with Qallow components
   - Connects to telemetry system
   - Hooks into memory system
   - Provides unified API

## Event Types

The system supports the following event types:

- **USER_FEEDBACK**: General user feedback
- **PERFORMANCE_ISSUE**: Performance problems reported
- **ERROR_REPORT**: Error and crash reports
- **FEATURE_REQUEST**: Feature requests
- **CONFIGURATION_CHANGE**: Configuration update requests
- **TELEMETRY_ANOMALY**: Anomalies detected in telemetry
- **MANUAL_TRIGGER**: Manual trigger for updates

## Quick Start

### Basic Usage

```python
from python.listeners import get_listener, submit_user_feedback, EventType

# Get the global listener
listener = get_listener()
listener.start()

# Submit feedback
submit_user_feedback(
    message="The system is running slowly",
    event_type=EventType.PERFORMANCE_ISSUE,
    metadata={"dataset_size": "1GB"},
    priority=3
)

# Check history
history = listener.get_event_history()
print(history)

listener.stop()
```

### With Integration

```python
from python.listeners import get_integration, EventType

# Get integration instance
integration = get_integration()
integration.start()

# Submit feedback through integration
integration.submit_feedback(
    message="Error: CUDA kernel failed",
    event_type=EventType.ERROR_REPORT,
    metadata={"error_code": 1},
    priority=5
)

# Get status
status = integration.get_status()
print(status)

integration.stop()
```

### Custom Callbacks

```python
from python.listeners import get_listener, EventType, UserEvent

listener = get_listener()

def my_callback(event: UserEvent):
    print(f"Event received: {event.message}")

listener.register_callback(EventType.USER_FEEDBACK, my_callback)
listener.start()
```

## Update Strategies

The AutoUpdater uses strategies to determine how to handle different event types:

### Configuration Updates
- Pattern: `set <key> = <value>`
- Action: Updates `qallow_config.json`
- Example: `set max_iterations = 1000`

### Performance Tuning
- Pattern: Keywords like "slow", "optimize", "performance"
- Action: Adjusts weights in `config/weights.json`
- Example: "The optimization is too slow"

### Error Patches
- Pattern: Keywords like "error", "fail", "crash"
- Action: Records error patch in `data/listeners/error_patches.json`
- Example: "Error: CUDA kernel failed"

## Data Storage

All listener data is stored in `data/listeners/`:

```
data/listeners/
├── listener.log              # Listener activity log
├── updater.log               # Updater activity log
├── integration.log           # Integration activity log
├── events_YYYYMMDD.jsonl     # Daily event history
├── updates_YYYYMMDD.jsonl    # Daily update history
├── telemetry_events.jsonl    # Telemetry events
├── error_log.jsonl           # Error log
└── anomalies.jsonl           # Anomaly detections
```

## API Reference

### UserListener

```python
class UserListener:
    def start() -> None
    def stop() -> None
    def submit_event(event: UserEvent) -> Dict
    def register_callback(event_type: EventType, callback: Callable) -> None
    def get_event_history(limit: int = 100) -> List[Dict]
    def get_update_history(limit: int = 100) -> List[Dict]
```

### AutoUpdater

```python
class AutoUpdater:
    def process_event(event: UserEvent) -> Optional[UpdateAction]
    def apply_update(action: UpdateAction) -> bool
    def get_update_history(limit: int = 100) -> List[Dict]
```

### QallowListenerIntegration

```python
class QallowListenerIntegration:
    def start() -> None
    def stop() -> None
    def submit_feedback(...) -> Dict
    def get_status() -> Dict[str, Any]
```

## Examples

Run the example script to see the system in action:

```bash
cd python/listeners
python example_usage.py
```

This will demonstrate:
1. Basic listener usage
2. Auto-updater with strategies
3. Full Qallow integration
4. Custom callback handling

## Integration with Qallow

The listener system integrates with:

- **Telemetry System**: Logs events to telemetry
- **Memory System**: Can store feedback in memory
- **Agent System**: Agents can submit feedback
- **Configuration System**: Updates configuration files
- **Error Tracking**: Records and analyzes errors

## Priority Levels

Events can have priority levels 1-5:

- **1**: Low priority, informational
- **2**: Normal priority
- **3**: Medium priority
- **4**: High priority, should be addressed
- **5**: Critical priority, immediate action needed

Higher priority events are processed first and may trigger automatic updates.

## Best Practices

1. **Use appropriate event types** for better categorization
2. **Include metadata** with relevant context
3. **Set priority correctly** to ensure timely processing
4. **Monitor logs** in `data/listeners/` for issues
5. **Test callbacks** before deploying to production
6. **Review update history** to understand system changes

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

## Future Enhancements

- Machine learning for pattern detection
- Predictive update recommendations
- Distributed listener across multiple nodes
- Web UI for monitoring and control
- Integration with CI/CD pipeline
- Automated testing of updates
- Rollback capabilities

## License

Part of the Qallow project. See LICENSE file for details.

