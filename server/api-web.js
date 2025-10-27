/**
 * Web App API Routes
 * Provides REST API endpoints for the React web application
 * Integrates with Qallow VM process management
 */

const express = require('express');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const router = express.Router();

// State management
let vmProcess = null;
let terminalOutput = [];
let metrics = {};
let auditLogs = [];
const MAX_OUTPUT_LINES = 1000;
const MAX_LOGS = 500;

// Logger
const logger = {
  info: (msg) => console.log(`[API] ${new Date().toISOString()} - ${msg}`),
  error: (msg, err) => {
    console.error(`[API ERROR] ${new Date().toISOString()} - ${msg}`);
    if (err) console.error(`  ${err.message}`);
  },
  success: (msg) => console.log(`[API SUCCESS] ${new Date().toISOString()} - ✅ ${msg}`)
};

// Helper to add terminal output
function addTerminalLine(content, type = 'info') {
  terminalOutput.push({
    timestamp: new Date().toISOString(),
    content,
    type
  });
  if (terminalOutput.length > MAX_OUTPUT_LINES) {
    terminalOutput.shift();
  }
}

// Helper to add audit log
function addAuditLog(component, message, level = 'Info') {
  auditLogs.push({
    timestamp: new Date().toISOString(),
    component,
    message,
    level
  });
  if (auditLogs.length > MAX_LOGS) {
    auditLogs.shift();
  }
}

// GET /api/status - Get current system status
router.get('/status', (req, res) => {
  try {
    res.json({
      vm_running: vmProcess !== null,
      terminal_output: terminalOutput,
      metrics: metrics,
      audit_logs: auditLogs,
      timestamp: new Date().toISOString()
    });
  } catch (err) {
    logger.error('Failed to get status', err);
    res.status(500).json({ error: err.message });
  }
});

// POST /api/vm/start - Start the Qallow VM
router.post('/vm/start', (req, res) => {
  try {
    if (vmProcess !== null) {
      return res.status(400).json({ error: 'VM already running' });
    }

    const ticks = req.body.ticks || 1000;
    const build = req.body.build || 'CPU';
    
    logger.info(`Starting Qallow VM (${build}, ticks: ${ticks})`);
    addTerminalLine(`🚀 Starting Qallow Unified System (${build} build, ticks: ${ticks})`, 'info');
    addAuditLog('VM', `Starting unified system with ${build} build`, 'Info');

    const qallowPath = '/root/Qallow/build/qallow';

    if (!fs.existsSync(qallowPath)) {
      const error = `Qallow executable not found at ${qallowPath}`;
      logger.error(error);
      addTerminalLine(`❌ ${error}`, 'error');
      addAuditLog('VM', error, 'Error');
      return res.status(500).json({ error });
    }

    // Use correct command format: qallow run unified (or qallow run pipeline)
    vmProcess = spawn(qallowPath, ['run', 'unified'], {
      stdio: ['ignore', 'pipe', 'pipe'],
      detached: false
    });

    // Handle stdout
    vmProcess.stdout.on('data', (data) => {
      const lines = data.toString().split('\n').filter(l => l.trim());
      lines.forEach(line => {
        addTerminalLine(line, 'success');
        logger.info(`VM: ${line}`);
      });
    });

    // Handle stderr
    vmProcess.stderr.on('data', (data) => {
      const lines = data.toString().split('\n').filter(l => l.trim());
      lines.forEach(line => {
        addTerminalLine(line, 'error');
        logger.error(`VM stderr: ${line}`);
      });
    });

    // Handle process exit
    vmProcess.on('exit', (code, signal) => {
      logger.info(`VM process exited with code ${code}, signal ${signal}`);
      addTerminalLine(`⏹️ VM process exited (code: ${code})`, 'warning');
      addAuditLog('VM', `Process exited with code ${code}`, 'Warning');
      vmProcess = null;
    });

    // Handle process error
    vmProcess.on('error', (err) => {
      logger.error('VM process error', err);
      addTerminalLine(`❌ VM error: ${err.message}`, 'error');
      addAuditLog('VM', `Process error: ${err.message}`, 'Error');
      vmProcess = null;
    });

    logger.success('VM started successfully');
    addTerminalLine('✅ VM started successfully', 'success');
    addAuditLog('VM', 'VM started successfully', 'Success');

    // Simulate metrics updates
    updateMetrics();

    res.json({
      success: true,
      message: 'VM started',
      timestamp: new Date().toISOString()
    });
  } catch (err) {
    logger.error('Failed to start VM', err);
    addTerminalLine(`❌ Failed to start VM: ${err.message}`, 'error');
    addAuditLog('VM', `Failed to start: ${err.message}`, 'Error');
    res.status(500).json({ error: err.message });
  }
});

// POST /api/vm/stop - Stop the Qallow VM
router.post('/vm/stop', (req, res) => {
  try {
    if (vmProcess === null) {
      return res.status(400).json({ error: 'VM not running' });
    }

    logger.info('Stopping VM process');
    addTerminalLine('⏹️ Stopping VM...', 'warning');
    addAuditLog('VM', 'Stopping VM', 'Warning');

    vmProcess.kill('SIGTERM');
    
    // Force kill after 5 seconds if still running
    setTimeout(() => {
      if (vmProcess !== null) {
        logger.warn('Force killing VM process');
        vmProcess.kill('SIGKILL');
      }
    }, 5000);

    vmProcess = null;

    logger.success('VM stopped');
    addTerminalLine('✅ VM stopped', 'success');
    addAuditLog('VM', 'VM stopped', 'Success');

    res.json({
      success: true,
      message: 'VM stopped',
      timestamp: new Date().toISOString()
    });
  } catch (err) {
    logger.error('Failed to stop VM', err);
    res.status(500).json({ error: err.message });
  }
});

// GET /api/metrics - Get current metrics
router.get('/metrics', (req, res) => {
  try {
    res.json({
      metrics: metrics,
      timestamp: new Date().toISOString()
    });
  } catch (err) {
    logger.error('Failed to get metrics', err);
    res.status(500).json({ error: err.message });
  }
});

// GET /api/logs - Get audit logs
router.get('/logs', (req, res) => {
  try {
    res.json({
      logs: auditLogs,
      count: auditLogs.length,
      timestamp: new Date().toISOString()
    });
  } catch (err) {
    logger.error('Failed to get logs', err);
    res.status(500).json({ error: err.message });
  }
});

// Helper to update metrics
function updateMetrics() {
  metrics = {
    fidelity: 0.85 + Math.random() * 0.15,
    energy: 0.5 + Math.random() * 0.5,
    risk: 0.1 + Math.random() * 0.2,
    reward: 0.7 + Math.random() * 0.3,
    coherence: 0.8 + Math.random() * 0.2,
    entanglement: 0.75 + Math.random() * 0.25
  };
}

// Periodic metrics update
setInterval(() => {
  if (vmProcess !== null) {
    updateMetrics();
  }
}, 2000);

module.exports = router;

