# Button Testing Guide ✅

## Overview

All buttons in the Qallow Native App are fully connected and functional. This guide shows how to test each button.

---

## Running the App

```bash
cd /root/Qallow
cargo run
```

The native desktop window will appear with all UI components.

---

## Button Testing Checklist

### 1. ▶️ Start VM Button

**What it does**: Starts the quantum VM with the selected build and phase

**How to test**:
1. Click the "Start VM" button
2. Check the Terminal tab for output: `🚀 Starting VM with CPU build on Phase13`
3. Check the Audit Log tab for: `VM started successfully`
4. Check `qallow.log` for: `✓ VM started successfully`

**Expected behavior**:
- ✅ Button becomes disabled while VM is running
- ✅ Terminal shows startup message
- ✅ Audit log records the event
- ✅ Log file is updated

---

### 2. ⏹️ Stop VM Button

**What it does**: Gracefully stops the running VM

**How to test**:
1. Start the VM first (see above)
2. Click the "Stop VM" button
3. Check the Terminal tab for output: `⏹️ VM stopped`
4. Check the Audit Log tab for: `VM stopped`

**Expected behavior**:
- ✅ Button becomes enabled after VM stops
- ✅ Terminal shows stop message
- ✅ Audit log records the event
- ✅ VM process is terminated gracefully

---

### 3. ⏸️ Pause Button

**What it does**: Pauses VM execution

**How to test**:
1. Start the VM first
2. Click the "Pause" button
3. Check the Terminal tab for output: `⏸️ VM paused`

**Expected behavior**:
- ✅ Terminal shows pause message
- ✅ Audit log records the event
- ✅ VM execution is paused

---

### 4. 🔄 Reset Button

**What it does**: Clears metrics and resets system state

**How to test**:
1. Stop the VM first (if running)
2. Click the "Reset" button
3. Check the Terminal tab for output: `🔄 System reset`
4. Check the Metrics tab - all values should be cleared

**Expected behavior**:
- ✅ Terminal shows reset message
- ✅ Audit log records the event
- ✅ Metrics are cleared
- ✅ State is reset to defaults

---

### 5. 📈 Export Metrics Button

**What it does**: Exports current metrics to JSON file

**How to test**:
1. Click the "Export Metrics" button
2. Check for file: `qallow_metrics_export.json`
3. Verify the file contains JSON metrics data

**Expected behavior**:
- ✅ File is created: `qallow_metrics_export.json`
- ✅ File contains valid JSON
- ✅ Console shows: `✓ Metrics exported to qallow_metrics_export.json`

---

### 6. 💾 Save Config Button

**What it does**: Saves current configuration to JSON file

**How to test**:
1. Click the "Save Config" button
2. Check for file: `qallow_phase_config.json`
3. Verify the file contains JSON configuration data

**Expected behavior**:
- ✅ File is created: `qallow_phase_config.json`
- ✅ File contains valid JSON
- ✅ Console shows success message

---

### 7. 📋 View Logs Button

**What it does**: Displays audit log entries

**How to test**:
1. Perform some actions (start VM, stop VM, etc.)
2. Click the "View Logs" button
3. Check the console output for audit log entries

**Expected behavior**:
- ✅ Console displays all audit log entries
- ✅ Each entry shows timestamp, level, component, and message
- ✅ Format: `[HH:MM:SS] Level - Component: Message`

---

### 8. 📦 Build Selection Dropdown

**What it does**: Selects CPU or CUDA build

**How to test**:
1. Click the Build dropdown
2. Select "CPU Build"
3. Check Terminal for: `📦 Build selected: CPU`
4. Select "CUDA Build"
5. Check Terminal for: `📦 Build selected: CUDA`

**Expected behavior**:
- ✅ Terminal shows build selection message
- ✅ Audit log records the selection
- ✅ Cannot change build while VM is running

---

### 9. 📍 Phase Selection Dropdown

**What it does**: Selects the quantum phase

**How to test**:
1. Click the Phase dropdown
2. Select "Phase 13"
3. Check Terminal for: `📍 Phase selected: Phase13`
4. Select "Phase 14"
5. Check Terminal for: `📍 Phase selected: Phase14`

**Expected behavior**:
- ✅ Terminal shows phase selection message
- ✅ Audit log records the selection
- ✅ Cannot change phase while VM is running

---

## Verification Files

After testing, check these files:

```bash
# Configuration file
cat qallow_config.json

# Application state
cat qallow_state.json

# Application logs
cat qallow.log

# Exported metrics
cat qallow_metrics_export.json

# Phase configuration
cat qallow_phase_config.json
```

---

## Error Handling

### Button Errors

If a button shows an error:
1. Check the console output for error message
2. Check `qallow.log` for detailed error information
3. Verify the application state (VM running, etc.)

### Common Issues

| Issue | Solution |
|-------|----------|
| Start VM fails | Check if VM already running |
| Stop VM fails | Check if VM is actually running |
| Reset fails | Stop VM first, then reset |
| Export fails | Check disk space and permissions |
| Save fails | Check disk space and permissions |

---

## Button Connection Architecture

```
Button Click
    ↓
FLTK Callback (main.rs)
    ↓
ButtonHandler Method (button_handlers.rs)
    ↓
Backend Operation (ProcessManager, etc.)
    ↓
State Update (AppState)
    ↓
UI Update (Terminal, Metrics, Audit Log)
    ↓
Logging (AppLogger)
    ↓
File Persistence (JSON files)
```

---

## Testing Automation

Run the integration tests:

```bash
cd /root/Qallow/native_app
cargo test --test button_integration_test
```

Expected output:
```
test result: ok. 32 passed; 0 failed
```

---

## Summary

✅ **All buttons are connected and functional**
✅ **All buttons have proper error handling**
✅ **All button actions are logged**
✅ **All state changes are persisted**
✅ **All UI updates are reflected**

---

**Status**: ✅ READY FOR TESTING
**Date**: 2025-10-25
**Version**: 1.0.0

