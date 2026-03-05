# Button Testing Guide - All Apps

## Quick Start

### Web App
```bash
cd /root/Qallow/web-app
npm install
npm start
# Opens at http://localhost:3000
```

### Native App
```bash
cd /root/Qallow/native_app
cargo build --release
cargo run --release
```

### Electron App
```bash
cd /root/Qallow/app
npm install
npm start
```

### Server (Backend)
```bash
cd /root/Qallow/server
npm install
npm start
# Runs on http://localhost:3001
```

## Web App Button Testing

### Control Panel Tab
1. **Start VM Button**
   - Click "▶️ Start VM"
   - Expected: VM starts with selected build and phase
   - Check: Terminal shows "[INFO] Starting Qallow Unified System..."

2. **Stop VM Button**
   - Click "⏹️ Stop VM" (only enabled when running)
   - Expected: VM stops gracefully
   - Check: Terminal shows "[INFO] VM stopped gracefully..."

3. **Phase Selection**
   - Select different phases (13-20)
   - Expected: Phase dropdown updates
   - Check: All phases show professional names

4. **Build Selection**
   - Toggle between CPU and CUDA
   - Expected: Build type changes
   - Check: Terminal shows build selection

5. **Export Metrics**
   - Click "📈 Export Metrics"
   - Expected: Metrics exported successfully
   - Check: Message shows "✅ Metrics exported successfully"

6. **Save Config**
   - Click "💾 Save Config"
   - Expected: Configuration saved
   - Check: Message shows "✅ Configuration saved successfully"

7. **View Logs**
   - Click "📋 View Logs"
   - Expected: Logs loaded
   - Check: Message shows log count

8. **Reset**
   - Click "🔄 Reset"
   - Expected: System resets
   - Check: Message shows "✅ System reset successfully"

## Native App Button Testing

### Control Panel Tab
1. **Start VM**
   - Click "▶️ Start"
   - Expected: VM starts with unified system
   - Check: Terminal shows startup message

2. **Stop VM**
   - Click "⏹️ Stop" (only enabled when running)
   - Expected: VM stops
   - Check: Terminal shows stop message

3. **Pause VM**
   - Click "⏸️ Pause" (only enabled when running)
   - Expected: VM pauses
   - Check: Terminal shows pause message with metrics

4. **Reset**
   - Click "🔄 Reset" (only enabled when stopped)
   - Expected: System resets
   - Check: Terminal shows reset message

5. **Phase Selection**
   - Select different phases (13-20)
   - Expected: Phase updates
   - Check: All phases available and professional names shown

6. **Build Selection**
   - Toggle CPU/CUDA
   - Expected: Build changes
   - Check: Terminal shows build selection

7. **Export Metrics**
   - Click "📈 Export Metrics"
   - Expected: Metrics exported
   - Check: Terminal shows export message

8. **Save Config**
   - Click "💾 Save Config"
   - Expected: Config saved to qallow_phase_config.json
   - Check: Terminal shows save message

9. **View Logs**
   - Click "📋 View Logs"
   - Expected: Audit logs displayed
   - Check: Terminal shows log entries

10. **Build Native App**
    - Click "🛠️ Build"
    - Expected: Build starts
    - Check: Terminal shows build progress

11. **Run Tests**
    - Click "🧪 Tests"
    - Expected: Tests run
    - Check: Terminal shows test results

12. **Git Status**
    - Click "📁 Git"
    - Expected: Git status shown
    - Check: Terminal shows git status

13. **Recent Commits**
    - Click "📜 Commits"
    - Expected: Recent commits listed
    - Check: Terminal shows commit history

## Electron App Button Testing

### Control Panel Tab
1. **Start VM**
   - Click "▶️ Start VM"
   - Expected: VM starts
   - Check: Status shows "🟢 Running"

2. **Stop VM**
   - Click "⏹️ Stop VM"
   - Expected: VM stops
   - Check: Status shows "🔴 Stopped"

3. **Phase Selection**
   - Select different phases (13-20)
   - Expected: Phase updates
   - Check: All phases available

4. **Ticks Configuration**
   - Adjust tick count
   - Expected: Ticks update
   - Check: Value changes

5. **Parameter Tuning**
   - Adjust target fidelity and epsilon
   - Expected: Parameters update
   - Check: Values change

## Cross-App Consistency Checks

### Phase Names
- [ ] Web app shows all 20 phases with professional names
- [ ] Native app shows all 20 phases with professional names
- [ ] Electron app shows all 20 phases with professional names
- [ ] All phase names match exactly

### Button Functionality
- [ ] Start button works in all apps
- [ ] Stop button works in all apps
- [ ] Phase selection works in all apps
- [ ] Build selection works in all apps
- [ ] Metrics export works in all apps
- [ ] Config save works in all apps

### Output Format
- [ ] All apps show professional output (no emoji)
- [ ] All apps use [INFO], [SUCCESS] format
- [ ] All apps show PASS/FAIL status
- [ ] All apps have consistent messaging

### Terminal Output
- [ ] Web app terminal shows all messages
- [ ] Native app terminal shows all messages
- [ ] Electron app terminal shows all messages
- [ ] All terminals have consistent formatting

## Verification Checklist

- [ ] All buttons are clickable
- [ ] All buttons execute their intended action
- [ ] All buttons show appropriate feedback
- [ ] All apps use professional terminology
- [ ] All apps support phases 13-20
- [ ] All apps have consistent UI/UX
- [ ] All apps produce professional output
- [ ] No gamified language in any app
- [ ] All error messages are clear
- [ ] All success messages are clear

## Known Issues & Fixes

### Issue: Phase selection not updating
**Fix**: Ensure phase choice menu is properly bound to callback

### Issue: Build selection not working
**Fix**: Verify build type enum is properly handled

### Issue: Buttons disabled when shouldn't be
**Fix**: Check VM running state logic

## Status

**TESTING READY** ✅

All three apps are ready for comprehensive button testing.

---

**Test Date**: 2025-10-28  
**Status**: Ready for Testing  
**Next Steps**: Execute all tests and verify functionality

