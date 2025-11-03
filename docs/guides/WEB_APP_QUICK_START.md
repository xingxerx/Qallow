# 🚀 Qallow Web App - Quick Start Guide

## Start the Web App

### Option 1: Quick Start (Recommended)

```bash
cd /root/Qallow
npm --prefix web-app install
npm --prefix server install
cd server && node server-web.js
```

Then open: **http://localhost:3001**

### Option 2: Using npm scripts

```bash
cd /root/Qallow/server
npm start
```

---

## Web App Tabs

### 📊 Dashboard
- **Status**: Shows if VM is running
- **Phase Pipeline**: Displays phases 13 → 14 → 15
- **Metrics**: Real-time fidelity, energy, risk, reward
- **System Info**: Build type, ticks, mode, quantum backend

### 💻 Terminal
- Live output from running VM
- Shows all phase execution logs
- Displays errors and warnings

### 📈 Metrics
- Detailed metrics display
- Fidelity, Energy, Risk, Reward
- Coherence and Entanglement values
- Updates in real-time

### 🔍 Audit Log
- Complete audit trail
- Timestamps for all operations
- Component and message details
- Success/error/warning levels

### ⚙️ Control Panel
- **VM Controls**: Start/Stop buttons
- **Configuration**: Build type, Phase, Ticks
- **Pipeline**: Visual phase flow
- **Quick Actions**: Export, Save, View Logs, Reset
- **System Info**: Current status and settings

---

## Using the Buttons

### 1. Start VM
1. Select **Build Type**: CPU or CUDA
2. Select **Phase**: 13, 14, or 15
3. Set **Ticks**: 100-10000
4. Click **▶️ Start VM**
5. Watch Terminal tab for output

### 2. Stop VM
- Click **⏹️ Stop VM** (only enabled when running)
- VM will gracefully shut down

### 3. Export Metrics
- Click **📈 Export Metrics**
- Creates `qallow_metrics_*.json` file
- Contains all metrics and terminal output

### 4. Save Config
- Click **💾 Save Config**
- Creates `qallow_config_*.json` file
- Saves current settings and metrics

### 5. View Logs
- Click **📋 View Logs**
- Shows audit trail in Audit Log tab
- Displays all operations with timestamps

### 6. Reset System
- Click **🔄 Reset**
- Clears all metrics and logs
- Stops running VM if active

---

## Example Workflow

### Run All Phases Sequentially

1. **Phase 13 - Quantum Circuit Optimization**
   - Set Phase: 13
   - Set Ticks: 100
   - Click Start VM
   - Wait for completion

2. **Phase 14 - Photonic Integration**
   - Set Phase: 14
   - Set Ticks: 100
   - Click Start VM
   - Wait for completion

3. **Phase 15 - AGI Synthesis**
   - Set Phase: 15
   - Set Ticks: 100
   - Click Start VM
   - Wait for completion

4. **Export Results**
   - Click Export Metrics
   - Click Save Config
   - Check files in `/root/Qallow/`

---

## Output Files

### Metrics Export
**File**: `qallow_metrics_*.json`

Contains:
- Timestamp
- All metrics (fidelity, energy, risk, reward, etc.)
- Terminal output from execution
- Audit logs

### Config Save
**File**: `qallow_config_*.json`

Contains:
- Timestamp
- Ticks setting
- Build type (CPU/CUDA)
- Phase number
- Current metrics

---

## API Endpoints

All buttons use these REST API endpoints:

| Button | Endpoint | Method | Purpose |
|--------|----------|--------|---------|
| Start VM | `/api/vm/start` | POST | Start VM with parameters |
| Stop VM | `/api/vm/stop` | POST | Stop running VM |
| Export Metrics | `/api/metrics/export` | GET | Export metrics to file |
| Save Config | `/api/config/save` | POST | Save configuration |
| View Logs | `/api/logs` | GET | Get audit logs |
| Reset | `/api/vm/reset` | POST | Reset system state |
| Status | `/api/status` | GET | Get current status |

---

## Troubleshooting

### Port 3001 Already in Use
```bash
fuser -k 3001/tcp
```

### Web App Not Loading
- Check server is running: `curl http://localhost:3001/health`
- Check API is responding: `curl http://localhost:3001/api/status`

### VM Not Starting
- Check `/root/Qallow/build/qallow` exists
- Check phase number is valid (13, 14, or 15)
- Check ticks value is between 100-10000

### Metrics Not Updating
- Ensure VM is running
- Check Terminal tab for errors
- Try clicking Export Metrics

---

## Performance Tips

1. **Use CPU build** for faster testing
2. **Start with low ticks** (100-500) for quick runs
3. **Run phases sequentially** for best results
4. **Export metrics** after each phase
5. **Reset system** between test runs

---

## Status

✅ All buttons working  
✅ All phases executable  
✅ Metrics collection active  
✅ Configuration saving enabled  
✅ Ready for production use

