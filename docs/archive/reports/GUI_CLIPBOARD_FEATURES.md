# Qallow GUI - Clipboard & Export Features

## Overview
The Qallow native GUI application now has full clipboard and export functionality to help you share and analyze output.

## Available Features

### 1. **Terminal Tab** 💻
- **Copy Button**: Copies all terminal output to system clipboard
  - Click "Copy" to copy the entire terminal buffer
  - A confirmation message appears: "Terminal output copied to clipboard"
  - Paste anywhere with Ctrl+V (or Cmd+V on Mac)

- **Clear Button**: Clears all terminal output
  - Removes all displayed logs from the terminal view

- **Export Button**: Saves terminal output to file
  - Exports to: `qallow_terminal_export.log`
  - Full timestamp and content preserved

### 2. **Audit Log Tab** 📋
- **Copy Button**: Copies all audit logs to clipboard
  - Includes timestamps, log levels, components, and messages
  - Useful for sharing system events and status changes

- **Clear Button**: Clears audit log display

- **Export Button**: Saves audit logs to file
  - Exports to: `qallow_audit_export.log`

- **Filter Dropdown**: Filter logs by level
  - All
  - Success
  - Info
  - Warning
  - Error

### 3. **Control Panel Tab** ⚙️
- **Export Metrics Button**: Exports performance metrics
  - Saves to: `qallow_metrics_export.json`
  - Includes fidelity, coherence, decoherence, and other metrics

- **Save Config Button**: Saves current configuration
  - Saves to: `qallow_phase_config.json`
  - Preserves phase selection, tick count, and other settings

- **View Logs Button**: Shows recent application logs
  - Displays up to 40 recent log entries

## How to Use

### To Share Terminal Output:
1. Run your Qallow phase (Phase 13, 14, or 15)
2. Click the **Copy** button in the Terminal tab
3. Paste the output in chat, email, or document with Ctrl+V

### To Export for Analysis:
1. Click the **Export** button in the Terminal tab
2. The file `qallow_terminal_export.log` is created in the current directory
3. Open with any text editor

### To Share Metrics:
1. Click **Export Metrics** in the Control Panel
2. The file `qallow_metrics_export.json` is created
3. Share the JSON file for analysis

## File Locations
All exported files are saved in the current working directory (typically `/root/Qallow`):
- `qallow_terminal_export.log` - Terminal output
- `qallow_audit_export.log` - Audit logs
- `qallow_metrics_export.json` - Performance metrics
- `qallow_phase_config.json` - Configuration settings

## Tips
- Use **Copy** for quick sharing in chat/messages
- Use **Export** for long-term storage and analysis
- Use **Filter** in Audit Log to focus on specific event types
- Check the confirmation messages to verify operations succeeded

