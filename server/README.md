# 🚀 Qallow Unified Server

Comprehensive server for managing frontend, backend, and quantum framework with robust error handling and monitoring.

## ✨ Features

- **Express.js Backend** - RESTful API for quantum algorithms
- **React Frontend** - Interactive dashboard for monitoring and control
- **WebSocket Support** - Real-time communication and updates
- **Error Handling** - Comprehensive error logging and recovery
- **Health Checks** - System monitoring and diagnostics
- **Circuit Breaker** - Fault tolerance pattern implementation
- **Quantum Integration** - Google Cirq framework integration
- **Production Ready** - Fully tested and documented

## 📋 Requirements

- Node.js >= 16.0.0
- npm >= 8.0.0
- Python 3.7+
- Google Cirq (installed via pip)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd /root/Qallow/server
npm install
```

### 2. Build Frontend

```bash
cd /root/Qallow/app
npm install
npm run build
```

### 3. Start Server

```bash
# Development mode
npm run dev

# Production mode
npm run prod

# Using startup script
bash start-server.sh
```

### 4. Access Dashboard

- **Server**: http://localhost:5000
- **API**: http://localhost:5000/api
- **WebSocket**: ws://localhost:5000

## 📡 API Endpoints

### Health & Status

```bash
GET /api/health
# Returns server health status

GET /api/quantum/status
# Returns quantum framework status

GET /api/system/metrics
# Returns system metrics (memory, CPU, uptime)
```

### Quantum Algorithms

```bash
POST /api/quantum/run-grover
# Run Grover's algorithm
# Body: { num_qubits: 3, target_state: 5 }

POST /api/quantum/run-bell-state
# Run Bell state test

POST /api/quantum/run-deutsch
# Run Deutsch algorithm

POST /api/quantum/run-all
# Run all algorithms
```

## 🛠️ Configuration

### Environment Variables

Create `.env` file in server directory:

```env
NODE_ENV=development
PORT=5000
FRONTEND_PORT=3000
LOG_LEVEL=info
QUANTUM_FRAMEWORK=cirq
```

### Server Configuration

Edit `server.js` to customize:
- Port and host settings
- CORS configuration
- Request size limits
- WebSocket settings

## 📊 Error Handling

The server includes comprehensive error handling:

### Error Handler Features

- **Structured Logging** - JSON formatted logs
- **File Persistence** - Logs saved to disk
- **In-Memory Buffer** - Last 1000 errors in memory
- **Statistics** - Error tracking and analysis
- **Log Rotation** - Automatic cleanup of old logs

### Usage

```javascript
const { ErrorHandler } = require('./errorHandler');
const errorHandler = new ErrorHandler();

// Log errors
errorHandler.error('Something went wrong', error, { context: 'data' });
errorHandler.warn('Warning message');
errorHandler.info('Info message');

// Get statistics
const stats = errorHandler.getStats();
console.log(stats);
```

## 🏥 Health Checks

The server includes health check system:

```javascript
const { HealthCheck } = require('./errorHandler');
const healthCheck = new HealthCheck();

// Register checks
healthCheck.register('database', async () => {
  // Check database connection
});

healthCheck.register('quantum', async () => {
  // Check quantum framework
});

// Run all checks
const results = await healthCheck.runAll();
```

## 🔌 Circuit Breaker

Fault tolerance pattern for external services:

```javascript
const { CircuitBreaker } = require('./errorHandler');
const breaker = new CircuitBreaker('quantum-api', 5, 60000);

// Execute with circuit breaker
try {
  const result = await breaker.execute(async () => {
    return await runQuantumAlgorithm();
  });
} catch (err) {
  console.error('Circuit breaker open:', err);
}
```

## 📈 Monitoring

### System Metrics

The server tracks:
- Memory usage (heap, external)
- CPU usage
- Uptime
- Request count
- Error count

### Access Metrics

```bash
curl http://localhost:5000/api/system/metrics
```

## 🧪 Testing

```bash
# Run tests
npm test

# Run with coverage
npm test -- --coverage

# Run specific test
npm test -- errorHandler.test.js
```

## 📝 Logging

Logs are stored in `/root/Qallow/logs/`:

- `qallow-YYYY-MM-DD.log` - Daily log files
- JSON format for easy parsing
- Automatic rotation after 7 days

### Log Levels

- `ERROR` - Critical errors
- `WARN` - Warnings
- `INFO` - General information
- `DEBUG` - Debug information

## 🔐 Security

- CORS enabled for frontend
- Request size limits (50MB)
- Error messages sanitized
- No sensitive data in logs
- Environment variables for secrets

## 🚀 Deployment

### Docker

```bash
docker build -t qallow-server .
docker run -p 5000:5000 qallow-server
```

### Kubernetes

```bash
kubectl apply -f k8s/qallow-server-deployment.yaml
```

### Systemd Service

```bash
sudo cp qallow-server.service /etc/systemd/system/
sudo systemctl enable qallow-server
sudo systemctl start qallow-server
```

## 📚 API Response Format

All API responses follow this format:

```json
{
  "success": true,
  "data": { /* response data */ },
  "error": null,
  "timestamp": "2025-10-25T12:00:00.000Z"
}
```

Error responses:

```json
{
  "success": false,
  "error": "Error message",
  "timestamp": "2025-10-25T12:00:00.000Z"
}
```

## 🐛 Troubleshooting

### Server won't start

1. Check if port is in use: `lsof -i :5000`
2. Check Node.js version: `node -v`
3. Check logs: `tail -f logs/qallow-*.log`

### Quantum algorithms fail

1. Check Cirq installation: `python3 -c "import cirq"`
2. Check Python path: `which python3`
3. Check quantum script: `ls quantum_algorithms/unified_quantum_framework_real_hardware.py`

### WebSocket connection fails

1. Check firewall settings
2. Check CORS configuration
3. Check browser console for errors

## 📞 Support

For issues or questions:
1. Check logs in `/root/Qallow/logs/`
2. Review error messages in dashboard
3. Check API health: `curl http://localhost:5000/api/health`

## 📄 License

MIT License - See LICENSE file

## 🎯 Status

✅ **PRODUCTION READY**

- All endpoints tested
- Error handling comprehensive
- Monitoring active
- Documentation complete

