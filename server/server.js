#!/usr/bin/env node

/**
 * QALLOW UNIFIED SERVER
 * Comprehensive server for managing frontend, backend, and quantum framework
 * Handles all internal errors and provides robust error handling
 */

const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const http = require('http');
const WebSocket = require('ws');

// Initialize Express app
const app = express();
const PORT = process.env.PORT || 5000;
const FRONTEND_PORT = process.env.FRONTEND_PORT || 3000;

// Middleware
app.use(cors());
app.use(bodyParser.json({ limit: '50mb' }));
app.use(bodyParser.urlencoded({ limit: '50mb', extended: true }));
app.use(express.static(path.join(__dirname, '../app/build')));

// Logger utility
const logger = {
  info: (msg) => console.log(`[INFO] ${new Date().toISOString()} - ${msg}`),
  error: (msg, err) => console.error(`[ERROR] ${new Date().toISOString()} - ${msg}`, err || ''),
  warn: (msg) => console.warn(`[WARN] ${new Date().toISOString()} - ${msg}`),
  debug: (msg) => console.log(`[DEBUG] ${new Date().toISOString()} - ${msg}`)
};

// Error handler middleware
const errorHandler = (err, req, res, next) => {
  logger.error('Unhandled error:', err);
  res.status(err.status || 500).json({
    success: false,
    error: err.message || 'Internal server error',
    timestamp: new Date().toISOString()
  });
};

// Health check endpoint
app.get('/api/health', (req, res) => {
  try {
    res.json({
      success: true,
      status: 'healthy',
      timestamp: new Date().toISOString(),
      uptime: process.uptime()
    });
  } catch (err) {
    logger.error('Health check failed:', err);
    res.status(500).json({ success: false, error: 'Health check failed' });
  }
});

// Quantum framework endpoints
app.post('/api/quantum/run-grover', async (req, res) => {
  try {
    const { num_qubits = 3, target_state = 5 } = req.body;
    logger.info(`Running Grover's algorithm with ${num_qubits} qubits, target ${target_state}`);
    
    const result = await runQuantumAlgorithm('grover', { num_qubits, target_state });
    res.json({ success: true, data: result });
  } catch (err) {
    logger.error('Grover algorithm failed:', err);
    res.status(500).json({ success: false, error: err.message });
  }
});

app.post('/api/quantum/run-bell-state', async (req, res) => {
  try {
    logger.info('Running Bell state test');
    const result = await runQuantumAlgorithm('bell', {});
    res.json({ success: true, data: result });
  } catch (err) {
    logger.error('Bell state failed:', err);
    res.status(500).json({ success: false, error: err.message });
  }
});

app.post('/api/quantum/run-deutsch', async (req, res) => {
  try {
    logger.info('Running Deutsch algorithm');
    const result = await runQuantumAlgorithm('deutsch', {});
    res.json({ success: true, data: result });
  } catch (err) {
    logger.error('Deutsch algorithm failed:', err);
    res.status(500).json({ success: false, error: err.message });
  }
});

// Run all quantum algorithms
app.post('/api/quantum/run-all', async (req, res) => {
  try {
    logger.info('Running all quantum algorithms');
    const results = {
      grover: await runQuantumAlgorithm('grover', { num_qubits: 3, target_state: 5 }),
      bell: await runQuantumAlgorithm('bell', {}),
      deutsch: await runQuantumAlgorithm('deutsch', {})
    };
    res.json({ success: true, data: results });
  } catch (err) {
    logger.error('All algorithms failed:', err);
    res.status(500).json({ success: false, error: err.message });
  }
});

// Get quantum framework status
app.get('/api/quantum/status', (req, res) => {
  try {
    res.json({
      success: true,
      framework: 'Google Cirq',
      simulator: 'QSim',
      status: 'ready',
      algorithms: ['Grover', 'Bell State', 'Deutsch'],
      timestamp: new Date().toISOString()
    });
  } catch (err) {
    logger.error('Status check failed:', err);
    res.status(500).json({ success: false, error: err.message });
  }
});

// System metrics endpoint
app.get('/api/system/metrics', (req, res) => {
  try {
    const metrics = {
      uptime: process.uptime(),
      memory: process.memoryUsage(),
      cpu: process.cpuUsage(),
      timestamp: new Date().toISOString()
    };
    res.json({ success: true, data: metrics });
  } catch (err) {
    logger.error('Metrics retrieval failed:', err);
    res.status(500).json({ success: false, error: err.message });
  }
});

// Serve React app for all other routes
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '../app/build/index.html'));
});

// Error handling middleware
app.use(errorHandler);

// Helper function to run quantum algorithms
async function runQuantumAlgorithm(algorithm, params) {
  return new Promise((resolve, reject) => {
    try {
      const pythonScript = path.join(__dirname, '../quantum_algorithms/unified_quantum_framework_real_hardware.py');
      
      if (!fs.existsSync(pythonScript)) {
        throw new Error('Quantum framework script not found');
      }

      const python = spawn('python3', [pythonScript]);
      let output = '';
      let errorOutput = '';

      python.stdout.on('data', (data) => {
        output += data.toString();
      });

      python.stderr.on('data', (data) => {
        errorOutput += data.toString();
      });

      python.on('close', (code) => {
        if (code !== 0) {
          reject(new Error(`Quantum algorithm failed: ${errorOutput}`));
        } else {
          resolve({ algorithm, output, timestamp: new Date().toISOString() });
        }
      });

      python.on('error', (err) => {
        reject(new Error(`Failed to spawn Python process: ${err.message}`));
      });
    } catch (err) {
      reject(err);
    }
  });
}

// Create HTTP server
const server = http.createServer(app);

// WebSocket setup for real-time updates
const wss = new WebSocket.Server({ server });

wss.on('connection', (ws) => {
  logger.info('WebSocket client connected');
  
  ws.on('message', (message) => {
    try {
      const data = JSON.parse(message);
      logger.debug(`WebSocket message: ${JSON.stringify(data)}`);
      
      // Echo back with timestamp
      ws.send(JSON.stringify({
        type: 'ack',
        data: data,
        timestamp: new Date().toISOString()
      }));
    } catch (err) {
      logger.error('WebSocket message error:', err);
      ws.send(JSON.stringify({ type: 'error', message: err.message }));
    }
  });

  ws.on('close', () => {
    logger.info('WebSocket client disconnected');
  });

  ws.on('error', (err) => {
    logger.error('WebSocket error:', err);
  });
});

// Start server
server.listen(PORT, () => {
  logger.info(`🚀 Qallow Server running on port ${PORT}`);
  logger.info(`📊 API available at http://localhost:${PORT}/api`);
  logger.info(`🌐 Frontend available at http://localhost:${PORT}`);
  logger.info(`✅ Server is ready to handle requests`);
});

// Graceful shutdown
process.on('SIGTERM', () => {
  logger.info('SIGTERM received, shutting down gracefully');
  server.close(() => {
    logger.info('Server closed');
    process.exit(0);
  });
});

process.on('SIGINT', () => {
  logger.info('SIGINT received, shutting down gracefully');
  server.close(() => {
    logger.info('Server closed');
    process.exit(0);
  });
});

// Unhandled promise rejection
process.on('unhandledRejection', (reason, promise) => {
  logger.error('Unhandled Rejection at:', promise, 'reason:', reason);
});

// Uncaught exception
process.on('uncaughtException', (err) => {
  logger.error('Uncaught Exception:', err);
  process.exit(1);
});

module.exports = app;

