# 🚀 Qallow Server - Quick Start Guide

## ✅ What's Fixed

- ✅ **NO Browser Opening** - Server runs in backend-only mode
- ✅ **Comprehensive Error Handling** - All errors caught and logged
- ✅ **Auto-Recovery** - Server continues running even on errors
- ✅ **Native App Integration** - Works with Rust/FLTK desktop app
- ✅ **Error Tracking** - API endpoints to view and clear errors

## 🚀 Quick Start

### Option 1: Backend Server Only (Fastest)
```bash
cd /root/Qallow/server
node server.js
```

### Option 2: With Startup Script
```bash
bash /root/Qallow/server/start-server.sh
```

### Option 3: With Native App (Full Experience)
```bash
bash /root/Qallow/START_NATIVE_APP.sh
```

## 📡 API Endpoints

### Health & Status
```bash
# Check server health
curl http://localhost:5000/api/health

# Get system metrics
curl http://localhost:5000/api/system/metrics

# Get quantum status
curl http://localhost:5000/api/quantum/status
```

### Error Management
```bash
# View all errors
curl http://localhost:5000/api/errors

# Clear error logs
curl -X POST http://localhost:5000/api/errors/clear
```

### Quantum Algorithms
```bash
# Run Grover's algorithm
curl -X POST http://localhost:5000/api/quantum/run-grover \
  -H "Content-Type: application/json" \
  -d '{"num_qubits": 3}'

# Run Bell state
curl -X POST http://localhost:5000/api/quantum/run-bell-state

# Run Deutsch algorithm
curl -X POST http://localhost:5000/api/quantum/run-deutsch

# Run all algorithms
curl -X POST http://localhost:5000/api/quantum/run-all
```

## 🛡️ Error Handling Features

1. **Try-Catch Blocks** - Every endpoint wrapped in error handling
2. **Error Recovery System** - Tracks last 100 errors with context
3. **Auto-Recovery** - Server doesn't exit on errors
4. **Error Logging** - All errors logged with timestamps
5. **Error API** - Retrieve and clear errors via API

## 📊 Logging

### Log Levels
- `[INFO]` - General information
- `[ERROR]` - Errors with details
- `[WARN]` - Warnings
- `[DEBUG]` - Debug information
- `[SUCCESS]` - Success messages

### View Logs
```bash
# Real-time logs
tail -f /root/Qallow/logs/server-*.log

# Or from console output when running server
```

## 🎯 Architecture

```
Native Desktop App (Rust/FLTK)
    ↓ REST API + WebSocket + IPC
Backend Server (Node.js)
    ↓ Error Handling & Recovery
Quantum Framework (Google Cirq)
```

## ✨ Key Features

- ✅ Backend-only (no web serving)
- ✅ NO browser opening
- ✅ Comprehensive error handling
- ✅ Auto-recovery enabled
- ✅ Error tracking API
- ✅ Health monitoring
- ✅ WebSocket support
- ✅ IPC socket support
- ✅ Graceful shutdown

## 🧪 Testing

```bash
# Test health check
curl http://localhost:5000/api/health

# Test with verbose output
curl -v http://localhost:5000/api/health

# Test with pretty JSON
curl http://localhost:5000/api/health | jq .
```

## 🔧 Configuration

Environment variables:
- `PORT` - Server port (default: 5000)
- `IPC_SOCKET` - IPC socket path (default: /tmp/qallow-backend.sock)
- `NODE_ENV` - Environment (development/production)

## 📝 Notes

- Server runs on `http://localhost:5000`
- WebSocket available at `ws://localhost:5000`
- IPC socket at `/tmp/qallow-backend.sock`
- Logs stored in `/root/Qallow/logs/`
- NO web browser will open
- Server handles all internal errors automatically

## 🎉 Ready to Go!

Your server is now ready to run with comprehensive error handling and auto-recovery!

```bash
cd /root/Qallow/server && node server.js
```

NO WEB BROWSER WILL OPEN!
