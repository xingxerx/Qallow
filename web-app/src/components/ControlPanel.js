import React, { useState } from 'react';
import axios from 'axios';
import './ControlPanel.css';

function ControlPanel({ vmRunning, onStart, onStop, loading }) {
  const [ticks, setTicks] = useState(1000);
  const [build, setBuild] = useState('CPU');
  const [phase, setPhase] = useState('13');
  const [executionMode, setExecutionMode] = useState('single'); // 'single' or 'unified'
  const [actionLoading, setActionLoading] = useState(false);
  const [actionMessage, setActionMessage] = useState('');

  const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:3001/api';

  const handleStartWithParams = async () => {
    try {
      if (executionMode === 'unified') {
        // Start continuous unified execution
        await axios.post(`${API_BASE}/vm/start-continuous`, {
          ticks,
          build,
          continuous: true
        });
      } else {
        // Start single phase
        await onStart({ ticks, build, phase });
      }
    } catch (error) {
      console.error('Error starting VM:', error);
    }
  };

  const handleExportMetrics = async () => {
    setActionLoading(true);
    try {
      await axios.get(`${API_BASE}/metrics/export`);
      setActionMessage('✅ Metrics exported successfully');
      setTimeout(() => setActionMessage(''), 3000);
    } catch (error) {
      setActionMessage('❌ Failed to export metrics');
      console.error('Error exporting metrics:', error);
    } finally {
      setActionLoading(false);
    }
  };

  const handleSaveConfig = async () => {
    setActionLoading(true);
    try {
      await axios.post(`${API_BASE}/config/save`, { ticks, build, phase });
      setActionMessage('✅ Configuration saved successfully');
      setTimeout(() => setActionMessage(''), 3000);
    } catch (error) {
      setActionMessage('❌ Failed to save configuration');
      console.error('Error saving config:', error);
    } finally {
      setActionLoading(false);
    }
  };

  const handleViewLogs = async () => {
    setActionLoading(true);
    try {
      const response = await axios.get(`${API_BASE}/logs`);
      setActionMessage(`✅ Loaded ${response.data.count} log entries`);
      setTimeout(() => setActionMessage(''), 3000);
    } catch (error) {
      setActionMessage('❌ Failed to load logs');
      console.error('Error loading logs:', error);
    } finally {
      setActionLoading(false);
    }
  };

  const handleReset = async () => {
    setActionLoading(true);
    try {
      await axios.post(`${API_BASE}/vm/reset`);
      setActionMessage('✅ System reset successfully');
      setTimeout(() => setActionMessage(''), 3000);
    } catch (error) {
      setActionMessage('❌ Failed to reset system');
      console.error('Error resetting system:', error);
    } finally {
      setActionLoading(false);
    }
  };

  return (
    <div className="control-panel">
      <div className="control-section">
        <h3>🎮 VM Controls</h3>
        <div className="button-group">
          <button
            className="btn btn-start"
            onClick={handleStartWithParams}
            disabled={vmRunning || loading}
          >
            {loading ? '⏳ Starting...' : '▶️ Start VM'}
          </button>
          <button
            className="btn btn-stop"
            onClick={onStop}
            disabled={!vmRunning || loading}
          >
            {loading ? '⏳ Stopping...' : '⏹️ Stop VM'}
          </button>
        </div>
      </div>

      <div className="control-section">
        <h3>⚙️ Configuration</h3>
        <div className="config-group">
          <div className="config-item">
            <label>Execution Mode</label>
            <select
              value={executionMode}
              onChange={(e) => setExecutionMode(e.target.value)}
              disabled={vmRunning}
              className="config-select"
            >
              <option value="single">Single Phase</option>
              <option value="unified">Unified (1→20 Loop)</option>
            </select>
          </div>

          <div className="config-item">
            <label>Build Type</label>
            <select
              value={build}
              onChange={(e) => setBuild(e.target.value)}
              disabled={vmRunning}
              className="config-select"
            >
              <option>CPU</option>
              <option>CUDA</option>
            </select>
          </div>

          <div className="config-item">
            <label>{executionMode === 'unified' ? 'Ticks per Phase' : 'Phase'}</label>
            {executionMode === 'unified' ? (
              <input
                type="number"
                value={ticks}
                onChange={(e) => setTicks(parseInt(e.target.value))}
                disabled={vmRunning}
                className="config-input"
                min="100"
                max="10000"
                step="100"
              />
            ) : (
              <select
                value={phase}
                onChange={(e) => setPhase(e.target.value)}
                disabled={vmRunning}
                className="config-select"
              >
                <option value="13">Phase 13 - Quantum Circuit Optimization</option>
                <option value="14">Phase 14 - Photonic Integration</option>
                <option value="15">Phase 15 - AGI Synthesis</option>
                <option value="16">Phase 16 - Constraint Validation</option>
                <option value="17">Phase 17 - State Persistence & Checkpointing</option>
                <option value="18">Phase 18 - Distributed Execution Coordinator</option>
                <option value="19">Phase 19 - Compliance Verification & Logging</option>
                <option value="20">Phase 20 - Result Synthesis & Aggregation</option>
              </select>
            )}
          </div>

          <div className="config-item">
            <label>Ticks</label>
            <input
              type="number"
              value={ticks}
              onChange={(e) => setTicks(parseInt(e.target.value))}
              disabled={vmRunning}
              className="config-input"
              min="100"
              max="10000"
              step="100"
            />
          </div>
        </div>
      </div>

      <div className="control-section">
        <h3>📊 Pipeline (20 Phases)</h3>
        <div className="pipeline-info">
          <div className="pipeline-stage">
            <div className="stage-number">13</div>
            <div className="stage-name">Quantum Opt</div>
          </div>
          <div className="pipeline-arrow">→</div>
          <div className="pipeline-stage">
            <div className="stage-number">14</div>
            <div className="stage-name">Photonic</div>
          </div>
          <div className="pipeline-arrow">→</div>
          <div className="pipeline-stage">
            <div className="stage-number">15</div>
            <div className="stage-name">AGI Synth</div>
          </div>
          <div className="pipeline-arrow">→</div>
          <div className="pipeline-stage">
            <div className="stage-number">16</div>
            <div className="stage-name">Constraint</div>
          </div>
          <div className="pipeline-arrow">→</div>
          <div className="pipeline-stage">
            <div className="stage-number">17</div>
            <div className="stage-name">Persistence</div>
          </div>
          <div className="pipeline-arrow">→</div>
          <div className="pipeline-stage">
            <div className="stage-number">18</div>
            <div className="stage-name">Distributed</div>
          </div>
          <div className="pipeline-arrow">→</div>
          <div className="pipeline-stage">
            <div className="stage-number">19</div>
            <div className="stage-name">Compliance</div>
          </div>
          <div className="pipeline-arrow">→</div>
          <div className="pipeline-stage">
            <div className="stage-number">20</div>
            <div className="stage-name">Synthesis</div>
          </div>
        </div>
      </div>

      <div className="control-section">
        <h3>📋 Quick Actions</h3>
        <div className="action-buttons">
          <button
            className="btn btn-secondary"
            onClick={handleExportMetrics}
            disabled={actionLoading}
          >
            {actionLoading ? '⏳' : '📈'} Export Metrics
          </button>
          <button
            className="btn btn-secondary"
            onClick={handleSaveConfig}
            disabled={actionLoading}
          >
            {actionLoading ? '⏳' : '💾'} Save Config
          </button>
          <button
            className="btn btn-secondary"
            onClick={handleViewLogs}
            disabled={actionLoading}
          >
            {actionLoading ? '⏳' : '📋'} View Logs
          </button>
          <button
            className="btn btn-secondary"
            onClick={handleReset}
            disabled={actionLoading}
          >
            {actionLoading ? '⏳' : '🔄'} Reset
          </button>
        </div>
        {actionMessage && (
          <div className="action-message">
            {actionMessage}
          </div>
        )}
      </div>

      <div className="control-section">
        <h3>ℹ️ System Info</h3>
        <div className="info-box">
          <div className="info-row">
            <span>Status:</span>
            <span className={vmRunning ? 'status-running' : 'status-stopped'}>
              {vmRunning ? '🟢 Running' : '🔴 Stopped'}
            </span>
          </div>
          <div className="info-row">
            <span>Build:</span>
            <span>{build}</span>
          </div>
          <div className="info-row">
            <span>Ticks:</span>
            <span>{ticks}</span>
          </div>
          <div className="info-row">
            <span>Mode:</span>
            <span>Unified System</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ControlPanel;

