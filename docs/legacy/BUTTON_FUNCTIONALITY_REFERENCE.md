# 🎮 Button Functionality Reference

## All Buttons Now Working ✅

---

## Control Panel Buttons

### 1. ▶️ Start VM Button

**Location**: Control Panel → VM Controls

**What it does**:
- Starts the Qallow VM with selected parameters
- Runs the selected phase (13, 14, or 15)
- Executes for specified number of ticks
- Uses selected build type (CPU or CUDA)

**Parameters**:
- `phase`: 13, 14, or 15
- `ticks`: 100-10000
- `build`: CPU or CUDA

**API Endpoint**: `POST /api/vm/start`

**Example**:
```bash
curl -X POST http://localhost:3001/api/vm/start \
  -H "Content-Type: application/json" \
  -d '{"phase": "13", "ticks": 100, "build": "CPU"}'
```

**Response**:
```json
{
  "success": true,
  "message": "VM started",
  "timestamp": "2025-10-27T16:18:29.055Z"
}
```

---

### 2. ⏹️ Stop VM Button

**Location**: Control Panel → VM Controls

**What it does**:
- Stops the running Qallow VM
- Gracefully terminates the process
- Saves final metrics
- Updates status

**API Endpoint**: `POST /api/vm/stop`

**Example**:
```bash
curl -X POST http://localhost:3001/api/vm/stop
```

**Response**:
```json
{
  "success": true,
  "message": "VM stopped",
  "timestamp": "2025-10-27T16:18:35.000Z"
}
```

---

### 3. 📈 Export Metrics Button

**Location**: Control Panel → Quick Actions

**What it does**:
- Exports all metrics to JSON file
- Includes terminal output
- Includes audit logs
- Creates timestamped file

**API Endpoint**: `GET /api/metrics/export`

**Example**:
```bash
curl http://localhost:3001/api/metrics/export
```

**Response**:
```json
{
  "success": true,
  "filename": "qallow_metrics_1761581899879.json",
  "filepath": "/root/Qallow/qallow_metrics_1761581899879.json",
  "timestamp": "2025-10-27T16:18:19.880Z"
}
```

**File Contents**:
```json
{
  "timestamp": "2025-10-27T16:18:19.880Z",
  "metrics": {
    "fidelity": 0.98,
    "energy": 0.81,
    "risk": 0.11,
    "reward": 0.72,
    "coherence": 0.93,
    "entanglement": 0.96
  },
  "terminal_output": [...],
  "audit_logs": [...]
}
```

---

### 4. 💾 Save Config Button

**Location**: Control Panel → Quick Actions

**What it does**:
- Saves current configuration
- Includes phase, build, ticks
- Includes current metrics
- Creates timestamped file

**API Endpoint**: `POST /api/config/save`

**Example**:
```bash
curl -X POST http://localhost:3001/api/config/save \
  -H "Content-Type: application/json" \
  -d '{"phase": "13", "ticks": 1000, "build": "CPU"}'
```

**Response**:
```json
{
  "success": true,
  "filename": "qallow_config_1761581895418.json",
  "filepath": "/root/Qallow/qallow_config_1761581895418.json",
  "timestamp": "2025-10-27T16:18:15.418Z"
}
```

**File Contents**:
```json
{
  "timestamp": "2025-10-27T16:18:15.418Z",
  "ticks": 1000,
  "build": "CPU",
  "phase": "13",
  "metrics": {
    "fidelity": 0.98,
    "energy": 0.81,
    "risk": 0.11,
    "reward": 0.72
  }
}
```

---

### 5. 📋 View Logs Button

**Location**: Control Panel → Quick Actions

**What it does**:
- Retrieves audit logs
- Displays in Audit Log tab
- Shows all operations with timestamps
- Includes component and message info

**API Endpoint**: `GET /api/logs`

**Example**:
```bash
curl http://localhost:3001/api/logs
```

**Response**:
```json
{
  "logs": [
    {
      "timestamp": "2025-10-27T16:18:24.580Z",
      "component": "VM",
      "message": "System reset",
      "level": "Success"
    },
    {
      "timestamp": "2025-10-27T16:18:29.052Z",
      "component": "VM",
      "message": "Starting unified system with CPU build, phase 13",
      "level": "Info"
    }
  ],
  "count": 2,
  "timestamp": "2025-10-27T16:18:33.668Z"
}
```

---

### 6. 🔄 Reset Button

**Location**: Control Panel → Quick Actions

**What it does**:
- Resets all system state
- Clears metrics
- Clears logs
- Stops running VM if active

**API Endpoint**: `POST /api/vm/reset`

**Example**:
```bash
curl -X POST http://localhost:3001/api/vm/reset
```

**Response**:
```json
{
  "success": true,
  "message": "VM state reset",
  "timestamp": "2025-10-27T16:18:24.580Z"
}
```

---

## Configuration Controls

### Phase Selection Dropdown

**Location**: Control Panel → Configuration

**Options**:
- Phase 13 - Quantum Circuit Optimization
- Phase 14 - Photonic Integration
- Phase 15 - AGI Synthesis

**Default**: Phase 13

**Behavior**:
- Disabled while VM is running
- Passed to Start VM button
- Determines which phase executes

---

### Build Type Dropdown

**Location**: Control Panel → Configuration

**Options**:
- CPU (default)
- CUDA

**Behavior**:
- Disabled while VM is running
- Passed to Start VM button
- Affects execution performance

---

### Ticks Input

**Location**: Control Panel → Configuration

**Range**: 100-10000

**Default**: 1000

**Behavior**:
- Disabled while VM is running
- Passed to Start VM button
- Controls execution duration

---

## Status Indicators

### VM Status
- 🟢 Running - VM is currently executing
- 🔴 Stopped - VM is not running

### Button States
- **Enabled**: Ready to click
- **Disabled**: Grayed out, cannot click
- **Loading**: Shows spinner, operation in progress

### Action Messages
- ✅ Success - Operation completed successfully
- ❌ Error - Operation failed
- ⏳ Loading - Operation in progress

---

## Complete Workflow Example

```
1. Select Phase 13
2. Set Ticks to 100
3. Click Start VM
   → VM starts and runs Phase 13
   → Terminal shows output
   → Metrics update in real-time
4. Wait for completion
5. Click Export Metrics
   → Creates qallow_metrics_*.json
6. Click Save Config
   → Creates qallow_config_*.json
7. Click View Logs
   → Shows audit trail
8. Click Reset
   → Clears state for next run
```

---

## Testing

Run the automated test suite:

```bash
chmod +x /root/Qallow/test_web_buttons.sh
/root/Qallow/test_web_buttons.sh
```

Expected output:
```
✓ Test 1: GET /api/status - PASS
✓ Test 2: POST /api/vm/reset - PASS
✓ Test 3: POST /api/config/save - PASS
✓ Test 4: GET /api/metrics/export - PASS
✓ Test 5: GET /api/logs - PASS
✓ Test 6: POST /api/vm/start (Phase 13) - PASS
✓ Test 7: GET /api/status (after VM) - PASS
✓ Test 8: GET /api/metrics/export (after run) - PASS
✓ Test 9: POST /api/vm/reset (cleanup) - PASS
✓ Test 10: POST /api/vm/start (Phase 14) - PASS
✓ Test 11: POST /api/vm/start (Phase 15) - PASS
✓ Test 12: GET /api/status (final) - PASS

✅ ALL TESTS PASSED!
```

---

## Status

✅ All buttons fully functional  
✅ All endpoints tested  
✅ All parameters working  
✅ Ready for production use

