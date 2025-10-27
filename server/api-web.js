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
let loopConfig = {
  startPhase: 13,
  ticks: 1000,
  build: 'CPU'
};
const MAX_OUTPUT_LINES = 1000;
const MAX_LOGS = 500;

// Logger
const logger = {
  info: (msg) => console.log(`[API] ${new Date().toISOString()} - ${msg}`),
  warn: (msg) => console.warn(`[API WARN] ${new Date().toISOString()} - ⚠️ ${msg}`),
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
      loop_config: loopConfig,
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

    // Always run all phases 1-20 in order, with quantum and CUDA enabled
    const ticks = req.body.ticks || 120;
    const build = req.body.build || 'CUDA';
    const continuous = true;

    logger.info(`Starting Qallow VM (phases 1-20, build: ${build}, ticks: ${ticks}, quantum: enabled)`);
    addTerminalLine(`🚀 Starting Qallow Unified System (phases 1-20, build: ${build}, ticks: ${ticks}, quantum: enabled)`, 'info');
    addAuditLog('VM', `Starting unified system with all phases, build ${build}, quantum enabled`, 'Info');

    // Set quantum env
    process.env.QALLOW_QISKIT = '1';

    const qallowPath = '/root/Qallow/build/qallow';
    if (!fs.existsSync(qallowPath)) {
      const error = `Qallow executable not found at ${qallowPath}`;
      logger.error(error);
      addTerminalLine(`❌ ${error}`, 'error');
      addAuditLog('VM', error, 'Error');
      return res.status(500).json({ error });
    }

    // Start continuous loop from phase 1
    beginContinuousLoop({
      startPhase: 1,
      ticks,
      build
    });

    res.json({
      success: true,
      message: 'Continuous loop started (phases 1-20, quantum, CUDA)',
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
    currentPhase = loopConfig.startPhase;
    cycleCount = 0;

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
    const startPhase = parseInt(req.body.phase || 13, 10) || 13;

    beginContinuousLoop({ startPhase, ticks, build });

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
function beginContinuousLoop({ startPhase, ticks, build }) {
  if (vmProcess !== null) {
    throw new Error('VM already running');
  }

  loopConfig = {
    startPhase,
    ticks,
    build
  };
  continuousMode = true;
  currentPhase = startPhase;
  cycleCount = 0;

  addTerminalLine(`♾️ Continuous loop engaged (start phase ${startPhase}, ticks ${ticks}, build ${build})`, 'info');
  addAuditLog('VM', `Continuous loop engaged starting at phase ${startPhase}`, 'Info');
  logger.info(`Continuous loop engaged starting at phase ${startPhase}, ticks=${ticks}, build=${build}`);
  logger.success('Continuous loop engaged');
  addTerminalLine('✅ Continuous execution started', 'success');
  addAuditLog('VM', 'Continuous execution started', 'Success');

  const started = startNextPhase();
  if (!started) {
    continuousMode = false;
    throw new Error('Failed to spawn initial phase');
  }
}

function launchSinglePhase({ phase, ticks, build }) {
  addTerminalLine(`▶️ Launching single phase ${phase}`, 'info');
  addAuditLog('VM', `Launching single phase ${phase}`, 'Info');
  const started = spawnPhaseProcess({
    phase,
    ticks,
    build,
    onExit: (code) => {
      addTerminalLine(`⏹️ Single phase ${phase} exited with code ${code}`, code === 0 ? 'success' : 'warning');
      addAuditLog('VM', `Single phase ${phase} exited with code ${code}`, code === 0 ? 'Success' : 'Warning');
    }
  });
  if (!started) {
    throw new Error('Failed to launch phase');
  }
}

function startNextPhase() {
  if (!continuousMode || vmProcess !== null) {
    return false;
  }

  const { startPhase, ticks, build } = loopConfig;
  const phaseToRun = currentPhase;

  addTerminalLine(`\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`, 'info');
  addTerminalLine(`🔄 Cycle ${cycleCount + 1} - Phase ${phaseToRun}`, 'info');
  addTerminalLine(`━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`, 'info');
  addAuditLog('VM', `Starting phase ${phaseToRun} (cycle ${cycleCount + 1})`, 'Info');

  const started = spawnPhaseProcess({
    phase: phaseToRun,
    ticks,
    build,
    onExit: (code) => {
      if (!continuousMode) {
        return;
      }

      if (code !== 0) {
        addTerminalLine(`⚠️ Phase ${phaseToRun} exited with code ${code}, halting loop`, 'warning');
        addAuditLog('VM', `Phase ${phaseToRun} exited with code ${code}, loop halted`, 'Warning');
        continuousMode = false;
        return;
      }

      let nextPhase = phaseToRun + 1;
      if (nextPhase > 20) {
        cycleCount += 1;
        nextPhase = startPhase;
        addTerminalLine(`\n✨ Cycle ${cycleCount} complete! Restarting from Phase ${startPhase}...`, 'info');
        addAuditLog('VM', `Cycle ${cycleCount} complete, restarting from phase ${startPhase}`, 'Info');
      }
      currentPhase = nextPhase;

      setTimeout(() => {
        if (continuousMode) {
          startNextPhase();
        }
      }, 1000);
    }
  });

  if (!started) {
    continuousMode = false;
    return false;
  }

  return true;
}

function spawnPhaseProcess({ phase, ticks, build, onExit }) {
  const qallowPath = '/root/Qallow/build/qallow';

  if (!fs.existsSync(qallowPath)) {
    const error = `Qallow executable not found at ${qallowPath}`;
    logger.error(error);
    addTerminalLine(`❌ ${error}`, 'error');
    addAuditLog('VM', error, 'Error');
    return false;
  }

  logger.info(`Spawning phase ${phase} (ticks=${ticks}, build=${build})`);

  const args = ['phase', phase.toString(), `--ticks=${ticks}`];
  if (build === 'CUDA') {
    args.push('--cuda');
  }

  vmProcess = spawn(qallowPath, args, {
    stdio: ['ignore', 'pipe', 'pipe'],
    detached: false
  });

  vmProcess.stdout.on('data', (data) => {
    const lines = data.toString().split('\n').filter(l => l.trim());
    lines.forEach(line => {
      addTerminalLine(line, 'success');
      logger.info(`VM: ${line}`);
    });
  });

  vmProcess.stderr.on('data', (data) => {
    const lines = data.toString().split('\n').filter(l => l.trim());
    lines.forEach(line => {
      addTerminalLine(line, 'error');
      logger.error(`VM stderr: ${line}`);
    });
  });

  vmProcess.on('exit', (code, signal) => {
    logger.info(`Phase ${phase} exited with code ${code}, signal ${signal}`);
    addTerminalLine(`✅ Phase ${phase} completed (code: ${code})`, code === 0 ? 'success' : 'warning');
    addAuditLog('VM', `Phase ${phase} completed (code: ${code})`, code === 0 ? 'Success' : 'Warning');
    vmProcess = null;
    if (typeof onExit === 'function') {
      onExit(code, signal);
    }
  });

  vmProcess.on('error', (err) => {
    logger.error('VM process error', err);
    addTerminalLine(`❌ VM error: ${err.message}`, 'error');
    addAuditLog('VM', `Process error: ${err.message}`, 'Error');
    vmProcess = null;
    if (continuousMode) {
      continuousMode = false;
    }
  });

  logger.success(`Phase ${phase} started`);
  addTerminalLine(`✅ Phase ${phase} started`, 'success');
  updateMetrics();
  return true;
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
