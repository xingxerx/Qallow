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
let continuousMode = false;
let currentPhase = 13;
let cycleCount = 0;
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
      vm_running: vmProcess !== null || continuousMode,
      continuous_mode: continuousMode,
      current_phase: currentPhase,
      cycle_count: cycleCount,
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
    const phase = req.body.phase || '13';

    logger.info(`Starting Qallow VM (${build}, phase: ${phase}, ticks: ${ticks})`);
    addTerminalLine(`🚀 Starting Qallow Unified System (${build} build, phase ${phase}, ticks: ${ticks})`, 'info');
    addAuditLog('VM', `Starting unified system with ${build} build, phase ${phase}`, 'Info');

    const qallowPath = '/root/Qallow/build/qallow';

    if (!fs.existsSync(qallowPath)) {
      const error = `Qallow executable not found at ${qallowPath}`;
      logger.error(error);
      addTerminalLine(`❌ ${error}`, 'error');
      addAuditLog('VM', error, 'Error');
      return res.status(500).json({ error });
    }

    // Use correct command format: qallow phase <N> with ticks
    const args = ['phase', phase, `--ticks=${ticks}`];
    if (build === 'CUDA') {
      args.push('--cuda');
    }

    vmProcess = spawn(qallowPath, args, {
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
    if (vmProcess === null && !continuousMode) {
      return res.status(400).json({ error: 'VM not running' });
    }

    logger.info('Stopping VM process');
    addTerminalLine('⏹️ Stopping VM...', 'warning');
    addAuditLog('VM', 'Stopping VM', 'Warning');

    // Stop continuous mode
    continuousMode = false;

    if (vmProcess !== null) {
      vmProcess.kill('SIGTERM');

      // Force kill after 5 seconds if still running
      setTimeout(() => {
        if (vmProcess !== null) {
          logger.warn('Force killing VM process');
          vmProcess.kill('SIGKILL');
        }
      }, 5000);

      vmProcess = null;
    }

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

// POST /api/vm/start-continuous - Start continuous unified phase execution
router.post('/vm/start-continuous', (req, res) => {
  try {
    if (vmProcess !== null) {
      return res.status(400).json({ error: 'VM already running' });
    }

    const ticks = req.body.ticks || 1000;
    const build = req.body.build || 'CPU';

    logger.info(`Starting continuous unified execution (${build}, ticks per phase: ${ticks})`);
    addTerminalLine(`🚀 Starting Continuous Unified Execution (${build} build, ${ticks} ticks per phase)`, 'info');
    addAuditLog('VM', `Starting continuous unified execution with ${build} build`, 'Info');

    continuousMode = true;
    currentPhase = 13;
    cycleCount = 0;

    // Start the first phase
    startNextPhase(ticks, build);

    res.json({
      success: true,
      message: 'Continuous execution started',
      timestamp: new Date().toISOString()
    });
  } catch (err) {
    logger.error('Failed to start continuous execution', err);
    addTerminalLine(`❌ Failed to start continuous execution: ${err.message}`, 'error');
    addAuditLog('VM', `Failed to start continuous: ${err.message}`, 'Error');
    res.status(500).json({ error: err.message });
  }
});

// Helper function to start next phase in continuous mode
function startNextPhase(ticks, build) {
  if (!continuousMode || vmProcess !== null) {
    return;
  }

  const qallowPath = '/root/Qallow/build/qallow';

  if (!fs.existsSync(qallowPath)) {
    const error = `Qallow executable not found at ${qallowPath}`;
    logger.error(error);
    addTerminalLine(`❌ ${error}`, 'error');
    addAuditLog('VM', error, 'Error');
    continuousMode = false;
    return;
  }

  logger.info(`Starting phase ${currentPhase} (cycle ${cycleCount + 1})`);
  addTerminalLine(`\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`, 'info');
  addTerminalLine(`🔄 Cycle ${cycleCount + 1} - Phase ${currentPhase}`, 'info');
  addTerminalLine(`━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`, 'info');
  addAuditLog('VM', `Starting phase ${currentPhase} (cycle ${cycleCount + 1})`, 'Info');

  const args = ['phase', currentPhase.toString(), `--ticks=${ticks}`];
  if (build === 'CUDA') {
    args.push('--cuda');
  }

  vmProcess = spawn(qallowPath, args, {
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

  // Handle process exit - move to next phase
  vmProcess.on('exit', (code, signal) => {
    logger.info(`Phase ${currentPhase} completed with code ${code}`);
    addTerminalLine(`✅ Phase ${currentPhase} completed (code: ${code})`, 'success');
    addAuditLog('VM', `Phase ${currentPhase} completed`, 'Success');
    vmProcess = null;

    if (continuousMode) {
      // Move to next phase
      currentPhase++;
      if (currentPhase > 15) {
        // Cycle complete, restart from phase 13
        currentPhase = 13;
        cycleCount++;
        addTerminalLine(`\n✨ Cycle ${cycleCount} complete! Restarting from Phase 13...`, 'info');
        addAuditLog('VM', `Cycle ${cycleCount} complete, restarting`, 'Info');
      }

      // Start next phase after a short delay
      setTimeout(() => {
        if (continuousMode) {
          startNextPhase(ticks, build);
        }
      }, 1000);
    }
  });

  // Handle process error
  vmProcess.on('error', (err) => {
    logger.error('VM process error', err);
    addTerminalLine(`❌ VM error: ${err.message}`, 'error');
    addAuditLog('VM', `Process error: ${err.message}`, 'Error');
    vmProcess = null;
    continuousMode = false;
  });

  logger.success(`Phase ${currentPhase} started`);
  addTerminalLine(`✅ Phase ${currentPhase} started`, 'success');
  updateMetrics();
}

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

// GET /api/metrics/export - Export metrics to file
router.get('/metrics/export', (req, res) => {
  try {
    const exportData = {
      timestamp: new Date().toISOString(),
      metrics: metrics,
      terminal_output: terminalOutput,
      audit_logs: auditLogs
    };

    const filename = `qallow_metrics_${Date.now()}.json`;
    const filepath = path.join('/root/Qallow', filename);

    fs.writeFileSync(filepath, JSON.stringify(exportData, null, 2));

    logger.success(`Metrics exported to ${filename}`);
    addAuditLog('Metrics', `Exported metrics to ${filename}`, 'Success');

    res.json({
      success: true,
      filename: filename,
      filepath: filepath,
      timestamp: new Date().toISOString()
    });
  } catch (err) {
    logger.error('Failed to export metrics', err);
    addAuditLog('Metrics', `Failed to export: ${err.message}`, 'Error');
    res.status(500).json({ error: err.message });
  }
});

// POST /api/config/save - Save configuration
router.post('/config/save', (req, res) => {
  try {
    const config = {
      timestamp: new Date().toISOString(),
      ticks: req.body.ticks || 1000,
      build: req.body.build || 'CPU',
      phase: req.body.phase || '13',
      metrics: metrics
    };

    const filename = `qallow_config_${Date.now()}.json`;
    const filepath = path.join('/root/Qallow', filename);

    fs.writeFileSync(filepath, JSON.stringify(config, null, 2));

    logger.success(`Configuration saved to ${filename}`);
    addAuditLog('Config', `Saved configuration to ${filename}`, 'Success');

    res.json({
      success: true,
      filename: filename,
      filepath: filepath,
      timestamp: new Date().toISOString()
    });
  } catch (err) {
    logger.error('Failed to save config', err);
    addAuditLog('Config', `Failed to save: ${err.message}`, 'Error');
    res.status(500).json({ error: err.message });
  }
});

// POST /api/vm/reset - Reset VM state
router.post('/vm/reset', (req, res) => {
  try {
    if (vmProcess !== null) {
      vmProcess.kill('SIGTERM');
      vmProcess = null;
    }

    // Reset state
    terminalOutput = [];
    metrics = {};
    auditLogs = [];

    logger.success('VM state reset');
    addAuditLog('VM', 'System reset', 'Success');

    res.json({
      success: true,
      message: 'VM state reset',
      timestamp: new Date().toISOString()
    });
  } catch (err) {
    logger.error('Failed to reset VM', err);
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

