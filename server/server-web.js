#!/usr/bin/env node

/**
 * QALLOW WEB APP SERVER
 * Backend API server for the React web application
 * Runs on port 3001 and provides REST API for VM management
 */

const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const http = require('http');
const WebSocket = require('ws');
const apiWeb = require('./api-web');

// Initialize Express app
const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(bodyParser.json({ limit: '50mb' }));
app.use(bodyParser.urlencoded({ limit: '50mb', extended: true }));

// Logger
const logger = {
  info: (msg) => console.log(`[WEB] ${new Date().toISOString()} - ${msg}`),
  error: (msg, err) => {
    console.error(`[WEB ERROR] ${new Date().toISOString()} - ${msg}`);
    if (err) console.error(`  ${err.message}`);
  },
  success: (msg) => console.log(`[WEB SUCCESS] ${new Date().toISOString()} - ✅ ${msg}`)
};

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    uptime: process.uptime()
  });
});

// Mount API routes
app.use('/api', apiWeb);

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    error: 'Not found',
    path: req.path,
    method: req.method
  });
});

// Error handler
app.use((err, req, res, next) => {
  logger.error('Unhandled error', err);
  res.status(500).json({
    error: err.message || 'Internal server error',
    timestamp: new Date().toISOString()
  });
});

// Create HTTP server
const server = http.createServer(app);

// WebSocket server for real-time updates
const wss = new WebSocket.Server({ server });

wss.on('connection', (ws) => {
  logger.info('WebSocket client connected');
  
  ws.on('message', (message) => {
    try {
      const data = JSON.parse(message);
      logger.info(`WebSocket message: ${data.type}`);
      
      ws.send(JSON.stringify({
        type: 'ack',
        data: data,
        timestamp: new Date().toISOString()
      }));
    } catch (err) {
      logger.error('WebSocket message error', err);
    }
  });

  ws.on('close', () => {
    logger.info('WebSocket client disconnected');
  });

  ws.on('error', (err) => {
    logger.error('WebSocket error', err);
  });
});

// Start server
server.listen(PORT, () => {
  logger.success(`╔════════════════════════════════════════════════════════════╗`);
  logger.success(`║  🌐 QALLOW WEB APP SERVER                                ║`);
  logger.success(`╚════════════════════════════════════════════════════════════╝`);
  logger.success(`Web API server running on http://localhost:${PORT}`);
  logger.success(`WebSocket available at ws://localhost:${PORT}`);
  logger.success(`React app connects to this server for VM management`);
  logger.success(`Status: Ready for web app connections`);
  logger.info('');
});

// Graceful shutdown
process.on('SIGTERM', () => {
  logger.info('SIGTERM received, shutting down gracefully...');
  server.close(() => {
    logger.info('Server closed');
    process.exit(0);
  });
});

process.on('SIGINT', () => {
  logger.info('SIGINT received, shutting down gracefully...');
  server.close(() => {
    logger.info('Server closed');
    process.exit(0);
  });
});

// Handle uncaught exceptions
process.on('uncaughtException', (err) => {
  logger.error('Uncaught exception', err);
});

// Handle unhandled promise rejections
process.on('unhandledRejection', (reason, promise) => {
  logger.error('Unhandled rejection', new Error(String(reason)));
});

module.exports = app;

