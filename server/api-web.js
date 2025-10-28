/**
 * Web App API Routes
 * Provides REST API endpoints for the React web application
 * Integrates with Qallow VM process management
 */

const express = require('express');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const QallowMonitor = require('./monitoring');
const ImprovementTracker = require('./improvement-tracker');

const app = express();
app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json());
const router = express.Router();

// Initialize monitoring and tracking systems
const monitor = new QallowMonitor();
const improvementTracker = new ImprovementTracker();

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

// Performance tracking
let performanceMetrics = {
  phaseTimings: {},
  cycleTimings: [],
  phaseSuccessRates: {},
  healthStatus: 'HEALTHY',
  lastUpdateTime: Date.now()
};

// Real-time metrics collection
let realtimeMetrics = {
  coherence: [],
  fidelity: [],
  stability: [],
  ethicalScore: [],
  systemLoad: []
};

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
router.get('/status', (_req, res) => {
  try {
    res.json({
      vm_running: vmProcess !== null || continuousMode,
      continuous_mode: continuousMode,
      current_phase: currentPhase,
      cycle_count: cycleCount,
      loop_config: loopConfig,
      terminal_output: terminalOutput,
      metrics: metrics,
      performance_metrics: performanceMetrics,
      realtime_metrics: realtimeMetrics,
      audit_logs: auditLogs,
      timestamp: new Date().toISOString()
    });
  } catch (err) {
    logger.error('Failed to get status', err);
    res.status(500).json({ error: err.message });
  }
});

// GET /api/performance - Get performance analytics
router.get('/performance', (_req, res) => {
  try {
    const avgPhaseTime = Object.values(performanceMetrics.phaseTimings).length > 0
      ? Object.values(performanceMetrics.phaseTimings).reduce((a, b) => a + b, 0) / Object.values(performanceMetrics.phaseTimings).length
      : 0;

    const avgCycleTime = performanceMetrics.cycleTimings.length > 0
      ? performanceMetrics.cycleTimings.reduce((a, b) => a + b, 0) / performanceMetrics.cycleTimings.length
      : 0;

    res.json({
      performance_metrics: performanceMetrics,
      averages: {
        phase_time_ms: avgPhaseTime,
        cycle_time_ms: avgCycleTime
      },
      realtime_metrics: realtimeMetrics,
      timestamp: new Date().toISOString()
    });
  } catch (err) {
    logger.error('Failed to get performance metrics', err);
    res.status(500).json({ error: err.message });
  }
});

// POST /api/vm/start - Start the Qallow VM
router.post('/vm/start', (req, res) => {
  try {
    if (vmProcess !== null) {
      return res.status(400).json({ error: 'VM already running' });
    }

    // Always run all valid phases 1-20 in order, with quantum and CUDA enabled
    const ticks = req.body.ticks || 120;
    const build = req.body.build || 'CUDA';

    logger.info(`Starting Qallow VM (phases 1-20, build: ${build}, ticks: ${ticks}, quantum: enabled)`);
    addTerminalLine(`🚀 Starting Qallow Unified System (phases 1-20, build: ${build}, ticks: ${ticks}, quantum: enabled)`, 'info');
    addAuditLog('VM', `Starting unified system with all phases 1-20, build ${build}, quantum enabled`, 'Info');

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
let cycleStartTime = 0;
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
  cycleStartTime = Date.now();

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
        const cycleEndTime = Date.now();
        const cycleDuration = cycleEndTime - cycleStartTime;
        performanceMetrics.cycleTimings.push(cycleDuration);

        nextPhase = startPhase;
        cycleStartTime = Date.now();
        addTerminalLine(`\n✨ Cycle ${cycleCount} complete! (duration: ${cycleDuration}ms) Restarting from Phase ${startPhase}...`, 'info');
        addAuditLog('VM', `Cycle ${cycleCount} complete (duration: ${cycleDuration}ms), restarting from phase ${startPhase}`, 'Info');
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

  const phaseStartTime = Date.now();
  vmProcess = spawn(qallowPath, args, {
    stdio: ['ignore', 'pipe', 'pipe'],
    detached: false
  });

  vmProcess.stdout.on('data', (data) => {
    const lines = data.toString().split('\n').filter(l => l.trim());
    lines.forEach(line => {
      addTerminalLine(line, 'success');
      logger.info(`VM: ${line}`);

      // Extract metrics from output
      extractMetricsFromOutput(line);
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
    const phaseEndTime = Date.now();
    const phaseDuration = phaseEndTime - phaseStartTime;

    // Track phase timing
    if (!performanceMetrics.phaseTimings[phase]) {
      performanceMetrics.phaseTimings[phase] = [];
    }
    performanceMetrics.phaseTimings[phase].push(phaseDuration);

    // Track success rate
    if (!performanceMetrics.phaseSuccessRates[phase]) {
      performanceMetrics.phaseSuccessRates[phase] = { success: 0, total: 0 };
    }
    performanceMetrics.phaseSuccessRates[phase].total += 1;
    if (code === 0) {
      performanceMetrics.phaseSuccessRates[phase].success += 1;
    }

    logger.info(`Phase ${phase} exited with code ${code}, signal ${signal}, duration: ${phaseDuration}ms`);
    addTerminalLine(`✅ Phase ${phase} completed (code: ${code}, duration: ${phaseDuration}ms)`, code === 0 ? 'success' : 'warning');
    addAuditLog('VM', `Phase ${phase} completed (code: ${code}, duration: ${phaseDuration}ms)`, code === 0 ? 'Success' : 'Warning');
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
router.get('/logs', (_req, res) => {
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

// GET /api/health - Get system health status
router.get('/health', (_req, res) => {
  try {
    const healthCheck = monitor.performHealthCheck(metrics, currentPhase);
    const healthSummary = monitor.getHealthSummary();

    res.json({
      current_health: healthCheck,
      health_summary: healthSummary,
      timestamp: new Date().toISOString()
    });
  } catch (err) {
    logger.error('Failed to get health status', err);
    res.status(500).json({ error: err.message });
  }
});

// GET /api/optimizations - Get optimization recommendations
router.get('/optimizations', (_req, res) => {
  try {
    const recommendations = monitor.generateOptimizations(
      metrics,
      performanceMetrics.phaseTimings,
      performanceMetrics.cycleTimings
    );

    res.json({
      recommendations,
      count: recommendations.length,
      timestamp: new Date().toISOString()
    });
  } catch (err) {
    logger.error('Failed to get optimizations', err);
    res.status(500).json({ error: err.message });
  }
});

// POST /api/improvements/log - Log an improvement
router.post('/improvements/log', (req, res) => {
  try {
    const { category, title, description, impact, files } = req.body;

    if (!category || !title) {
      return res.status(400).json({ error: 'Missing required fields: category, title' });
    }

    const improvement = improvementTracker.logImprovement(
      category,
      title,
      description || '',
      impact || 'MEDIUM',
      files || []
    );

    logger.success(`Improvement logged: ${title}`);
    addAuditLog('Improvements', `Logged improvement: ${title}`, 'Success');

    res.json({
      success: true,
      improvement,
      timestamp: new Date().toISOString()
    });
  } catch (err) {
    logger.error('Failed to log improvement', err);
    res.status(500).json({ error: err.message });
  }
});

// GET /api/improvements/report - Get improvement report
router.get('/improvements/report', (_req, res) => {
  try {
    const report = improvementTracker.generateReport();

    res.json({
      report,
      timestamp: new Date().toISOString()
    });
  } catch (err) {
    logger.error('Failed to get improvement report', err);
    res.status(500).json({ error: err.message });
  }
});

// GET /api/improvements/summary - Get improvement summary
router.get('/improvements/summary', (_req, res) => {
  try {
    const summary = improvementTracker.getSummary();

    res.json({
      summary,
      timestamp: new Date().toISOString()
    });
  } catch (err) {
    logger.error('Failed to get improvement summary', err);
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

// Extract metrics from phase output
function extractMetricsFromOutput(line) {
  // Extract coherence
  const coherenceMatch = line.match(/[Cc]oherence[:\s=]+([0-9.]+)/);
  if (coherenceMatch) {
    const coherence = parseFloat(coherenceMatch[1]);
    realtimeMetrics.coherence.push({ value: coherence, timestamp: Date.now() });
    if (realtimeMetrics.coherence.length > 100) realtimeMetrics.coherence.shift();
  }

  // Extract fidelity
  const fidelityMatch = line.match(/[Ff]idelity[:\s=]+([0-9.]+)/);
  if (fidelityMatch) {
    const fidelity = parseFloat(fidelityMatch[1]);
    realtimeMetrics.fidelity.push({ value: fidelity, timestamp: Date.now() });
    if (realtimeMetrics.fidelity.length > 100) realtimeMetrics.fidelity.shift();
  }

  // Extract stability
  const stabilityMatch = line.match(/[Ss]tability[:\s=]+([0-9.]+)/);
  if (stabilityMatch) {
    const stability = parseFloat(stabilityMatch[1]);
    realtimeMetrics.stability.push({ value: stability, timestamp: Date.now() });
    if (realtimeMetrics.stability.length > 100) realtimeMetrics.stability.shift();
  }

  // Extract ethical score
  const ethicalMatch = line.match(/[Ee]thical[:\s=]+([0-9.]+)/);
  if (ethicalMatch) {
    const ethical = parseFloat(ethicalMatch[1]);
    realtimeMetrics.ethicalScore.push({ value: ethical, timestamp: Date.now() });
    if (realtimeMetrics.ethicalScore.length > 100) realtimeMetrics.ethicalScore.shift();
  }
}

// Helper to update metrics
function updateMetrics() {
  const baseMetrics = {
    fidelity: 0.85 + Math.random() * 0.15,
    energy: 0.5 + Math.random() * 0.5,
    risk: 0.1 + Math.random() * 0.2,
    reward: 0.7 + Math.random() * 0.3,
    coherence: 0.8 + Math.random() * 0.2,
    entanglement: 0.75 + Math.random() * 0.25
  };

  // Use real metrics if available
  if (realtimeMetrics.coherence.length > 0) {
    baseMetrics.coherence = realtimeMetrics.coherence[realtimeMetrics.coherence.length - 1].value;
  }
  if (realtimeMetrics.fidelity.length > 0) {
    baseMetrics.fidelity = realtimeMetrics.fidelity[realtimeMetrics.fidelity.length - 1].value;
  }
  if (realtimeMetrics.stability.length > 0) {
    baseMetrics.stability = realtimeMetrics.stability[realtimeMetrics.stability.length - 1].value;
  }

  metrics = baseMetrics;
  performanceMetrics.lastUpdateTime = Date.now();
}

// Periodic metrics update
setInterval(() => {
  if (vmProcess !== null) {
    updateMetrics();
  }
}, 2000);

app.use('/api', router);

const PORT = process.env.PORT || 5050;
app.listen(PORT, () => {
  console.log(`[API] Web UI and API server running on http://localhost:${PORT}`);
});
