# Qallow Web Application Guide

## Overview

The Qallow Web Application is a modern React-based interface for managing and monitoring the Qallow Unified Quantum-AGI System. It provides real-time visualization of system metrics, terminal output, and control capabilities.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    React Web App (Port 3000)                │
│  ┌──────────────┬──────────────┬──────────────────────────┐ │
│  │  Dashboard   │  Terminal    │  Metrics  │  Audit Log   │ │
│  │  Control     │  Panel       │           │              │ │
│  └──────────────┴──────────────┴──────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            ↓ HTTP/WebSocket
┌─────────────────────────────────────────────────────────────┐
│              Web API Server (Port 3001)                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  REST API Routes                                     │   │
│  │  - /api/status      (GET)                            │   │
│  │  - /api/vm/start    (POST)                           │   │
│  │  - /api/vm/stop     (POST)                           │   │
│  │  - /api/metrics     (GET)                            │   │
│  │  - /api/logs        (GET)                            │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↓ Process Management
┌─────────────────────────────────────────────────────────────┐
│         Qallow VM (/root/Qallow/build/qallow)               │
│  Unified System: Phase 13 → Phase 14 → Phase 15             │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Option 1: Using the Startup Script (Recommended)

```bash
cd /root/Qallow
./start-web-app.sh
```

This will:
1. Start the Web API Server on port 3001
2. Start the React Frontend on port 3000
3. Open the web app in your browser

### Option 2: Manual Startup

**Terminal 1 - Start API Server:**
```bash
cd /root/Qallow/server
node server-web.js
```

**Terminal 2 - Start React App:**
```bash
cd /root/Qallow/web-app
npm start
```

Then open http://localhost:3000 in your browser.

## Features

### 📊 Dashboard
- Real-time system status
- Key metrics display (Fidelity, Energy, Risk, Reward)
- Phase pipeline visualization
- System information

### 💻 Terminal
- Live output streaming from Qallow VM
- Color-coded log levels (Info, Success, Warning, Error)
- Auto-scrolling to latest output
- Line count display

### 📈 Metrics
- Detailed performance analytics
- Visual progress bars
- Status indicators
- Metric history

### 🔍 Audit Log
- Event tracking and filtering
- Component-based logging
- Timestamp tracking
- Level-based filtering (All, Info, Success, Warning, Error)

### ⚙️ Control Panel
- Start/Stop VM buttons
- Build type selection (CPU/CUDA)
- Ticks configuration
- Quick action buttons
- System information display

## API Endpoints

### GET /api/status
Returns current system status, terminal output, metrics, and audit logs.

**Response:**
```json
{
  "vm_running": true,
  "terminal_output": [...],
  "metrics": {...},
  "audit_logs": [...],
  "timestamp": "2025-10-27T14:00:00.000Z"
}
```

### POST /api/vm/start
Starts the Qallow VM with unified system.

**Request Body:**
```json
{
  "ticks": 1000,
  "build": "CPU"
}
```

### POST /api/vm/stop
Stops the running Qallow VM.

### GET /api/metrics
Returns current system metrics.

### GET /api/logs
Returns audit logs.

## File Structure

```
/root/Qallow/
├── web-app/                    # React frontend
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── components/
│   │   │   ├── Dashboard.js
│   │   │   ├── Terminal.js
│   │   │   ├── Metrics.js
│   │   │   ├── AuditLog.js
│   │   │   └── ControlPanel.js
│   │   ├── App.js
│   │   ├── App.css
│   │   ├── index.js
│   │   └── index.css
│   └── package.json
├── server/
│   ├── server-web.js           # Web API server
│   ├── api-web.js              # API routes
│   └── package.json
└── start-web-app.sh            # Startup script
```

## Configuration

### Environment Variables

**Web App (.env in web-app/):**
```
REACT_APP_API_URL=http://localhost:3001/api
```

**API Server (.env in server/):**
```
PORT=3001
```

## Styling

The web app uses a cyberpunk-inspired dark theme:
- **Primary Color:** `#00d4ff` (Cyan)
- **Accent Color:** `#00ff64` (Green)
- **Background:** `#0a0e27` (Dark Blue)
- **Secondary:** `#1a1f3a` (Navy)

## Troubleshooting

### Port Already in Use
If port 3000 or 3001 is already in use:

```bash
# Kill process on port 3000
lsof -ti:3000 | xargs kill -9

# Kill process on port 3001
lsof -ti:3001 | xargs kill -9
```

### API Connection Error
Ensure the API server is running on port 3001:
```bash
curl http://localhost:3001/health
```

### VM Not Starting
Check that the Qallow executable exists:
```bash
ls -lh /root/Qallow/build/qallow
```

If missing, rebuild:
```bash
cd /root/Qallow/build
cmake --build . --target qallow
```

## Development

### Install Dependencies
```bash
cd /root/Qallow/web-app
npm install
```

### Run Development Server
```bash
npm start
```

### Build for Production
```bash
npm run build
```

## Performance

- **Polling Interval:** 1 second (status updates)
- **Metrics Update:** 2 seconds
- **Max Terminal Lines:** 1000
- **Max Audit Logs:** 500

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

## License

MIT
