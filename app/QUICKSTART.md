# 🚀 Qallow Unified App - Quick Start

Get the Qallow Unified App running in 5 minutes!

## Prerequisites

- Node.js 16+ installed
- npm or yarn
- Qallow built (run `cd /root/Qallow && cmake -B build && cd build && make`)

## Installation (5 minutes)

### Step 1: Navigate to app directory
```bash
cd /root/Qallow/app
```

### Step 2: Install dependencies
```bash
npm install
```

### Step 3: Start the app
```bash
npm run electron-dev
```

That's it! The app should open automatically.

## What You'll See

### Dashboard Tab
- **Overlay Stability**: Real-time metrics for all 4 overlay types
- **Ethics Monitoring**: Safety, Clarity, and Human scores
- **Coherence**: Quantum coherence tracking
- **System Status**: All components at a glance

### Metrics Tab
- Performance metrics (throughput, latency, GPU utilization)
- Memory usage (GPU and CPU)
- Network statistics
- Auto-refreshes every 5 seconds

### Terminal Tab
- Live output from Qallow VM
- Real-time error logging
- Timestamp for each line

### Audit Log Tab
- Event logging with filtering
- Filter by log level (INFO, SUCCESS, WARNING, ERROR)
- Component-based tracking

### Control Panel Tab
- Start/Stop VM buttons
- Phase configuration
- Parameter adjustment
- Quick actions

## Common Tasks

### Start the Qallow VM
1. Go to **Control Panel** tab
2. Click **▶️ Start VM** button
3. Watch the Terminal tab for output

### Run a specific phase
1. Go to **Control Panel** tab
2. Select phase from dropdown
3. Adjust ticks and parameters
4. Click **▶️ Run Phase**

### View real-time metrics
1. Go to **Metrics** tab
2. Metrics auto-refresh every 5 seconds
3. Click **🔄 Refresh** for immediate update

### Check system events
1. Go to **Audit Log** tab
2. Use filter buttons to filter by level
3. Scroll through event history

### Stop the VM
1. Go to **Control Panel** tab
2. Click **⏹️ Stop VM** button

## Keyboard Shortcuts

- `Ctrl+Q` - Quit app
- `Ctrl+Shift+I` - Open developer tools (dev mode)
- `F5` - Reload app

## Troubleshooting

### App won't start
```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
npm run electron-dev
```

### Qallow VM won't start
```bash
# Check if binary exists
ls -la /root/Qallow/build/qallow_unified

# Make it executable
chmod +x /root/Qallow/build/qallow_unified

# Check CUDA
nvidia-smi
```

### Metrics not loading
```bash
# Start the web dashboard in another terminal
cd /root/Qallow/ui
python3 dashboard.py
```

### Port already in use
```bash
# Kill process on port 5000
lsof -ti:5000 | xargs kill -9

# Or use different port
PORT=5001 python3 dashboard.py
```

## Next Steps

1. **Explore the Dashboard** - Get familiar with real-time metrics
2. **Run a Phase** - Use Control Panel to run Phase 14 or 15
3. **Monitor Execution** - Watch Terminal and Metrics tabs
4. **Check Audit Log** - Review system events
5. **Export Data** - Use Quick Actions to export metrics

## Tips & Tricks

- **Dark Theme**: App uses dark theme optimized for long sessions
- **Responsive**: Works on different screen sizes
- **Real-time**: All metrics update in real-time
- **Filtering**: Use Audit Log filters to find specific events
- **Parameters**: Adjust phase parameters before running

## Performance

- Lightweight React app (~5MB)
- Efficient rendering
- Low CPU usage
- Responsive UI

## Support

For more information:
- Read `/root/Qallow/app/README.md` for detailed documentation
- Check `/root/Qallow/docs/` for Qallow documentation
- Review `/root/Qallow/ui/WEB_DASHBOARD_README.md` for web dashboard

## Production Build

When ready to deploy:
```bash
npm run electron-build
```

This creates a packaged app in `dist/` directory.

---

**Happy quantum computing! 🚀**

