# Quick Start Guide - All Apps

## One-Command Setup

### Terminal 1: Start Backend Server
```bash
cd /root/Qallow/server && npm install && npm start
# Server runs on http://localhost:3001
```

### Terminal 2: Start Web App
```bash
cd /root/Qallow/web-app && npm install && npm start
# Web app opens at http://localhost:3000
```

### Terminal 3: Start Native App
```bash
cd /root/Qallow/native_app && cargo build --release && cargo run --release
# Native app launches in FLTK window
```

### Terminal 4: Start Electron App
```bash
cd /root/Qallow/app && npm install && npm start
# Electron app launches in desktop window
```

## Individual App Quick Start

### Web App Only
```bash
cd /root/Qallow/web-app
npm install
npm start
# Opens at http://localhost:3000
```

### Native App Only
```bash
cd /root/Qallow/native_app
cargo build --release
cargo run --release
```

### Electron App Only
```bash
cd /root/Qallow/app
npm install
npm start
```

### Server Only
```bash
cd /root/Qallow/server
npm install
npm start
# Runs on http://localhost:3001
```

## Testing All Buttons

### Web App Buttons
1. Open http://localhost:3000
2. Go to "Control Panel" tab
3. Test buttons:
   - ▶️ Start VM
   - ⏹️ Stop VM
   - 📈 Export Metrics
   - 💾 Save Config
   - 📋 View Logs
   - 🔄 Reset

### Native App Buttons
1. Launch native app
2. Go to "Control Panel" tab
3. Test buttons:
   - ▶️ Start
   - ⏹️ Stop
   - ⏸️ Pause
   - 🔄 Reset
   - ⏭️ Advance
   - 🎚️ Tempo
   - 📈 Export Metrics
   - 💾 Save Config
   - 📋 View Logs
   - 🛠️ Build
   - 🧪 Tests
   - 📁 Git
   - 📜 Commits

### Electron App Buttons
1. Launch Electron app
2. Go to "Control Panel" tab
3. Test buttons:
   - ▶️ Start VM
   - ⏹️ Stop VM
   - Phase selection
   - Ticks configuration
   - Parameter tuning

## Phase Selection

All apps support phases 13-20:

- **Phase 13**: Quantum Circuit Optimization
- **Phase 14**: Photonic Integration
- **Phase 15**: AGI Synthesis
- **Phase 16**: Constraint Validation
- **Phase 17**: State Persistence & Checkpointing
- **Phase 18**: Distributed Execution Coordinator
- **Phase 19**: Compliance Verification & Logging
- **Phase 20**: Result Synthesis & Aggregation

## Build Selection

All apps support two build types:
- **CPU**: Standard CPU-based execution
- **CUDA**: GPU-accelerated execution

## Expected Output

### Professional Output Format
```
[INFO] Initializing synthesis vector...
[INFO] Creating superposition of result states...
[SUCCESS] Phase 20 Complete: Result synthesis finished
```

### Terminal Messages
- `[INFO]` - Information messages
- `[SUCCESS]` - Successful operations
- `[WARNING]` - Warning messages
- `[ERROR]` - Error messages

## Troubleshooting

### Web App Won't Start
```bash
cd /root/Qallow/web-app
rm -rf node_modules package-lock.json
npm install
npm start
```

### Native App Won't Compile
```bash
cd /root/Qallow/native_app
cargo clean
cargo build --release
```

### Electron App Won't Start
```bash
cd /root/Qallow/app
rm -rf node_modules package-lock.json
npm install
npm start
```

### Server Won't Start
```bash
cd /root/Qallow/server
rm -rf node_modules package-lock.json
npm install
npm start
```

### Port Already in Use
- Web app (3000): `lsof -i :3000` then `kill -9 <PID>`
- Server (3001): `lsof -i :3001` then `kill -9 <PID>`

## Verification Checklist

- [ ] Server starts without errors
- [ ] Web app loads at http://localhost:3000
- [ ] Native app launches in FLTK window
- [ ] Electron app launches in desktop window
- [ ] All phase options visible (13-20)
- [ ] All buttons are clickable
- [ ] Professional output shown (no emoji)
- [ ] Build selection works (CPU/CUDA)
- [ ] Metrics export works
- [ ] Config save works
- [ ] Logs display correctly

## Performance Tips

1. **Use Release Builds**
   ```bash
   cargo build --release  # Native app
   npm run build          # Web/Electron apps
   ```

2. **Monitor Resources**
   - Check CPU usage: `top`
   - Check memory: `free -h`
   - Check disk: `df -h`

3. **Optimize Settings**
   - Reduce ticks for faster execution
   - Use CPU build for testing
   - Use CUDA build for production

## Next Steps

1. Start all apps using the commands above
2. Test all buttons (see BUTTON_TESTING_GUIDE.md)
3. Verify cross-app consistency
4. Deploy to production

## Support

For issues or questions:
1. Check BUTTON_TESTING_GUIDE.md
2. Check APP_SYNC_FINAL_REPORT.md
3. Review application logs
4. Check server logs

---

**Last Updated**: 2025-10-28  
**Status**: Ready for Use  
**All Apps**: Synchronized and Tested

