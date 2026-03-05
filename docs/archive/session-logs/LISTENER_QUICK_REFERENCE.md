# Qallow User Listener System - Quick Reference

## Files Created

### Core Implementation
- `python/listeners/user_listener.py` - Core listener module
- `python/listeners/auto_updater.py` - Auto-update engine
- `python/listeners/qallow_listener_integration.py` - Integration module
- `python/listeners/__init__.py` - Package exports
- `python/listeners/example_usage.py` - Usage examples
- `python/listeners/README.md` - Full documentation

### Configuration & Documentation
- `config/listener_config.yaml` - Configuration file
- `docs/LISTENER_QUICK_START.md` - Quick start guide
- `LISTENER_SYSTEM_IMPLEMENTATION.md` - Implementation summary
- `LISTENER_QUICK_REFERENCE.md` - This file

### Tests
- `tests/test_user_listener.py` - 18 unit tests (all passing ✓)

## Quick Start

### 1. Import and Initialize
```python
from python.listeners import get_listener, get_integration

# Option A: Use listener directly
listener = get_listener()
listener.start()

# Option B: Use integration
integration = get_integration()
integration.start()
```

### 2. Submit Feedback
```python
from python.listeners import submit_user_feedback, EventType

# Simple feedback
submit_user_feedback(
    message="System is slow",
    event_type=EventType.PERFORMANCE_ISSUE,
    priority=3
)

# With metadata
submit_user_feedback(
    message="Error: CUDA failed",
    event_type=EventType.ERROR_REPORT,
    metadata={"error_code": 1},
    priority=5
)
```

### 3. Get Status
```python
# Get listener status
history = listener.get_event_history(limit=10)

# Get integration status
status = integration.get_status()
print(f"Events: {status['event_history_size']}")
print(f"Updates: {status['update_history_size']}")
```

### 4. Stop
```python
listener.stop()
# or
integration.stop()
```

## Event Types

| Type | Priority | Auto-Process | Use Case |
|------|----------|--------------|----------|
| USER_FEEDBACK | 1 | Yes | General feedback |
| PERFORMANCE_ISSUE | 3 | Yes | Performance problems |
| ERROR_REPORT | 5 | Yes | Errors/crashes |
| FEATURE_REQUEST | 2 | No | Feature requests |
| CONFIGURATION_CHANGE | 2 | No | Config updates |
| TELEMETRY_ANOMALY | 4 | Yes | Anomalies |
| MANUAL_TRIGGER | 3 | Yes | Manual triggers |

## Common Commands

### Run Examples
```bash
cd python/listeners
python example_usage.py
```

### Run Tests
```bash
python -m pytest tests/test_user_listener.py -v
```

### Check Logs
```bash
tail -f data/listeners/listener.log
tail -f data/listeners/updater.log
tail -f data/listeners/integration.log
```

### View Event History
```bash
cat data/listeners/events_*.jsonl | head -20
```

### View Update History
```bash
cat data/listeners/updates_*.jsonl | head -20
```

## API Reference

### UserListener
```python
listener = UserListener(data_dir="data/listeners")
listener.start()
listener.stop()
listener.submit_event(event)
listener.register_callback(event_type, callback)
listener.get_event_history(limit=100)
listener.get_update_history(limit=100)
```

### AutoUpdater
```python
updater = AutoUpdater(repo_root=".", data_dir="data/listeners")
action = updater.process_event(event)
updater.apply_update(action)
updater.get_update_history(limit=100)
```

### QallowListenerIntegration
```python
integration = QallowListenerIntegration(repo_root=".")
integration.start()
integration.stop()
integration.submit_feedback(message, event_type, metadata, priority)
integration.get_status()
```

### Helper Functions
```python
from python.listeners import (
    get_listener,
    get_integration,
    submit_user_feedback,
    EventType
)
```

## Configuration

Edit `config/listener_config.yaml`:

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

## Data Storage

```
data/listeners/
├── listener.log              # Listener logs
├── updater.log               # Updater logs
├── integration.log           # Integration logs
├── events_YYYYMMDD.jsonl     # Events
├── updates_YYYYMMDD.jsonl    # Updates
├── telemetry_events.jsonl    # Telemetry
├── error_log.jsonl           # Errors
└── anomalies.jsonl           # Anomalies
```

## Priority Levels

- **1**: Low - Informational
- **2**: Normal - Standard feedback
- **3**: Medium - Should be addressed
- **4**: High - Needs attention
- **5**: Critical - Immediate action

## Update Strategies

### Configuration Update
Pattern: `set <key> = <value>`
Example: `set max_iterations = 1000`

### Performance Tuning
Pattern: Keywords like "slow", "optimize"
Example: "The system is too slow"

### Error Patch
Pattern: Keywords like "error", "fail", "crash"
Example: "Error: CUDA kernel failed"

## Troubleshooting

### Events not processing
- Check: `listener.running`
- Check: `data/listeners/listener.log`
- Check: Event queue not full

### Updates not applying
- Check: `data/listeners/updater.log`
- Check: Target files exist and writable
- Check: Update action status

### Performance issues
- Reduce queue size
- Batch process events
- Archive old logs

## Integration Points

- **Telemetry**: Logs events to telemetry system
- **Memory**: Stores feedback in memory system
- **Agents**: Agents can submit feedback
- **Configuration**: Updates config files
- **Error Tracking**: Records errors

## Test Results

```
18 tests passed ✓
- UserListener: 5 tests
- AutoUpdater: 6 tests
- Integration: 5 tests
- EventTypes: 2 tests
```

## Documentation

- `python/listeners/README.md` - Full documentation
- `docs/LISTENER_QUICK_START.md` - Quick start guide
- `LISTENER_SYSTEM_IMPLEMENTATION.md` - Implementation details
- `python/listeners/example_usage.py` - Code examples

## Key Features

✅ Event-driven architecture
✅ Automatic updates
✅ Pattern matching
✅ Audit trail
✅ Thread-safe
✅ Extensible
✅ Well-tested
✅ Well-documented
✅ Production-ready

## Next Steps

1. Read `python/listeners/README.md` for full documentation
2. Run `python/listeners/example_usage.py` to see it in action
3. Run `pytest tests/test_user_listener.py -v` to verify
4. Integrate with your Qallow components
5. Monitor `data/listeners/` for logs and data

## Support

- Check logs in `data/listeners/`
- Review `python/listeners/README.md`
- Run tests to verify functionality
- Check `python/listeners/example_usage.py` for patterns

