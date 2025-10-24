# 🚀 Qallow Unified App

A modern desktop application for the Qallow Quantum-Photonic AGI system, providing a unified interface for all components: Terminal, Collector, AGI Agent, Quantum Algorithms, and Circuit Simulation.

## Features

### 📊 Dashboard
- Real-time overlay stability monitoring (Orbital, River, Mycelial, Global)
- Ethics monitoring with Safety, Clarity, and Human scores
- Coherence tracking with decoherence measurements
- Execution status and GPU acceleration info
- System component status overview

### 📈 Metrics
- Real-time performance metrics
- Phase status monitoring
- GPU and CPU memory usage
- Network statistics
- Auto-refresh every 5 seconds

### 💻 Terminal
- Live terminal output from Qallow VM
- Real-time error logging
- Timestamp tracking
- Scrollable output history

### 🔍 Audit Log
- Comprehensive event logging
- Filterable by log level (INFO, SUCCESS, WARNING, ERROR)
- Component-based filtering
- Timestamp tracking

### ⚙️ Control Panel
- Start/Stop VM controls
- Phase configuration
- Parameter adjustment
- Quick actions (Export, Save, Reset)
- System information display

## Installation

### Prerequisites
- Node.js 16+ and npm
- Electron 27+
- Qallow build directory with compiled binaries

### Setup

```bash
cd /root/Qallow/app

# Install dependencies
npm install

# For development with hot reload
npm run electron-dev

# For production build
npm run electron-build
```

## Usage

### Development Mode
```bash
npm run electron-dev
```
This starts both the React dev server and Electron app with hot reload.

### Production Build
```bash
npm run electron-build
```
Creates a packaged application in the `dist/` directory.

### Web Dashboard (Alternative)
If you prefer a web-based interface:
```bash
cd /root/Qallow/ui
python3 dashboard.py
# Open http://localhost:5000
```

## Architecture

### Components

1. **Dashboard** - Real-time system metrics and status
2. **Metrics** - Performance and resource monitoring
3. **Terminal** - Live output from Qallow VM
4. **Audit Log** - Event logging and filtering
5. **Control Panel** - System control and configuration

### Technology Stack

- **Frontend**: React 18, CSS3
- **Desktop**: Electron 27
- **Backend**: Node.js with IPC
- **Build**: Webpack (via react-scripts)

## File Structure

```
app/
├── public/
│   └── index.html
├── src/
│   ├── components/
│   │   ├── Dashboard.js
│   │   ├── Dashboard.css
│   │   ├── Terminal.js
│   │   ├── Terminal.css
│   │   ├── Metrics.js
│   │   ├── Metrics.css
│   │   ├── AuditLog.js
│   │   ├── AuditLog.css
│   │   ├── ControlPanel.js
│   │   └── ControlPanel.css
│   ├── App.js
│   ├── App.css
│   ├── index.js
│   └── index.css
├── main.js
├── preload.js
├── package.json
└── README.md
```

## Configuration

### Electron Configuration (package.json)
- Window size: 1600x1000 (minimum 1200x800)
- Dev tools enabled in development
- Auto-reload on file changes

### IPC Handlers
- `start-qallow` - Start the Qallow VM
- `stop-qallow` - Stop the Qallow VM
- `get-metrics` - Fetch current metrics
- `get-audit-logs` - Fetch audit logs

## Styling

The app uses a modern dark theme with cyan accents:
- Primary color: #00d4ff (Cyan)
- Background: #0a0e27 (Dark blue)
- Accent: #00ff64 (Green for success)
- Error: #ff6464 (Red)

## Performance

- Lightweight React components
- Efficient re-rendering with hooks
- Lazy loading of metrics
- Optimized CSS with gradients
- Responsive design for all screen sizes

## Troubleshooting

### App won't start
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
npm run electron-dev
```

### Qallow process not starting
- Ensure `/root/Qallow/build/qallow_unified` exists
- Check file permissions: `chmod +x /root/Qallow/build/qallow_unified`
- Verify CUDA is installed: `nvidia-smi`

### Metrics not loading
- Ensure web dashboard is running: `cd ui && python3 dashboard.py`
- Check if port 5000 is available
- Verify network connectivity

## Development

### Adding a new component
1. Create component file in `src/components/`
2. Create corresponding CSS file
3. Import in `App.js`
4. Add navigation button in sidebar

### Modifying styles
- Global styles: `src/App.css`, `src/index.css`
- Component styles: `src/components/*.css`
- Use CSS variables for consistency

## Building for Distribution

```bash
# Create production build
npm run electron-build

# Output will be in dist/ directory
# Supports Windows, macOS, and Linux
```

## License

MIT - See LICENSE file in root directory

## Support

For issues or questions:
1. Check the Qallow documentation: `/root/Qallow/docs/`
2. Review the web dashboard: `/root/Qallow/ui/`
3. Check system logs: `/root/Qallow/data/logs/`

## Version

- App Version: 1.0.0
- Qallow Version: 1.0.0
- Electron: 27.0.0
- React: 18.2.0

