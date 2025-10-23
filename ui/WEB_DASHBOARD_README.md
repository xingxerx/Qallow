# 🎯 Qallow Web Dashboard - Enhanced User Interface

**Lightweight, Cross-Platform Web-Based Monitoring for Quantum-Photonic AGI System**

A Flask-based real-time dashboard providing comprehensive visualization of Qallow system metrics, ethics monitoring, and phase progression.

## ✨ Features

### 🧠 Cognitive State Monitoring
- Real-time reward, energy, and risk metrics
- Step-by-step progression tracking
- Visual progress bars for quick assessment
- Module count and telemetry point tracking

### 📊 Phase Status Dashboard
- Current phase identification
- Phase progress tracking
- Fidelity and coherence metrics
- Real-time phase metrics from CSV logs
- Multi-phase comparison

### ⚖️ Ethics Monitoring
- Safety (S), Clarity (C), and Human (H) scores
- Total ethics score (E = S + C + H)
- Visual progress indicators
- Compliance tracking

### 📋 Audit Log Viewer
- Real-time ethics audit log display
- Last 50 audit entries
- Searchable and filterable logs
- Timestamp tracking

### 📈 Telemetry Visualization
- Reward trajectory chart
- Energy & risk analysis
- Historical data retention (1000 points)
- Real-time chart updates using Chart.js

### 🔄 Phase Metrics Integration
- Automatic CSV telemetry loading from `data/logs/`
- JSON metrics parsing
- Multi-phase comparison
- Performance trending

## 🚀 Quick Start

### Installation

```bash
cd /root/Qallow/ui
pip install -r requirements.txt
```

### Running the Dashboard

```bash
# Start the Flask server
python3 dashboard.py

# Access the dashboard
# Open browser to: http://localhost:5000
```

### Docker Deployment

```bash
docker build -t qallow-dashboard .
docker run -p 5000:5000 \
  -v /root/Qallow/data:/root/Qallow/data \
  qallow-dashboard
```

## 📡 API Endpoints

### `/api/state`
Get current system state

```bash
curl http://localhost:5000/api/state
```

Response:
```json
{
  "reward": 0.127,
  "energy": 0.376,
  "risk": 0.339,
  "modules": 18,
  "step": 50,
  "running": true,
  "current_phase": "phase13",
  "fidelity": 0.981,
  "coherence": 0.999
}
```

### `/api/telemetry`
Get telemetry history (last 1000 points)

```bash
curl http://localhost:5000/api/telemetry
```

### `/api/ethics`
Get ethics scores

```bash
curl http://localhost:5000/api/ethics
```

### `/api/phases`
Get phase metrics from CSV logs

```bash
curl http://localhost:5000/api/phases
```

### `/api/audit`
Get ethics audit log entries (last 50)

```bash
curl http://localhost:5000/api/audit
```

### `/api/start`
Start the mind process

```bash
curl http://localhost:5000/api/start
```

### `/api/stop`
Stop the mind process

```bash
curl http://localhost:5000/api/stop
```

## 📁 Data Sources

The dashboard automatically integrates with:

- **CSV Telemetry**: `data/logs/phase*.csv`
- **JSON Metrics**: `data/logs/phase*.json`
- **Audit Logs**: `data/ethics_audit.log`
- **Real-time Output**: Mind process stdout

## ⚙️ Configuration

### Environment Variables

```bash
export FLASK_ENV=production
export FLASK_DEBUG=0
export QALLOW_DATA_DIR=/root/Qallow/data
```

### Port Configuration

Edit `dashboard.py` line 230:
```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

## 📊 Performance

- **Update Interval**: 500ms
- **Telemetry Buffer**: 1000 points
- **Audit Log**: Last 50 entries
- **Phase Metrics**: Last 10 entries per phase
- **Memory Usage**: ~50MB typical

## 🔧 Troubleshooting

### Dashboard not loading
```bash
# Check Flask is running
curl http://localhost:5000/

# Check logs
tail -f /tmp/qallow_dashboard.log
```

### No telemetry data
```bash
# Verify CSV files exist
ls -la /root/Qallow/data/logs/phase*.csv

# Check file permissions
chmod 644 /root/Qallow/data/logs/phase*.csv
```

### Audit log empty
```bash
# Verify audit log file
ls -la /root/Qallow/data/ethics_audit.log

# Check file permissions
chmod 644 /root/Qallow/data/ethics_audit.log
```

## 🌐 Browser Compatibility

- Chrome/Chromium 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## 🔒 Security

- CORS enabled for local development
- Restrict to localhost in production
- Use reverse proxy (nginx) for external access
- Implement authentication for sensitive data

## 📚 Integration with Qallow

### Running Phases with Dashboard

```bash
# Terminal 1: Start dashboard
cd /root/Qallow/ui
python3 dashboard.py

# Terminal 2: Run phase
cd /root/Qallow
./build/qallow phase 13 --ticks=400 --log=data/logs/phase13.csv

# Terminal 3: Monitor in browser
# Open http://localhost:5000
```

### Viewing Results

1. **Real-time Metrics**: Dashboard updates every 500ms
2. **Phase Completion**: Check "Phase Status" card
3. **Ethics Compliance**: Monitor "Ethics Monitoring" card
4. **Audit Trail**: Review "Ethics Audit Log" section
5. **Historical Data**: Charts show full trajectory

## 📈 Development

### Adding New Metrics

1. Add API endpoint in `dashboard.py`
2. Add data loading function
3. Update HTML template with new card
4. Add JavaScript update function

### Customizing Appearance

Edit `templates/dashboard.html` CSS section:
```css
/* Colors */
--primary: #00ff88;
--secondary: #00ffff;
--background: #0a0e27;
```

## 📝 License

MIT - See LICENSE file

## 🤝 Support

For issues or feature requests, see the main Qallow documentation.

