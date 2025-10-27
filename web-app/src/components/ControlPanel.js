import React, { useState } from 'react';
import './ControlPanel.css';

function ControlPanel({ vmRunning, onStart, onStop, loading }) {
  const [ticks, setTicks] = useState(1000);
  const [build, setBuild] = useState('CPU');

  return (
    <div className="control-panel">
      <div className="control-section">
        <h3>🎮 VM Controls</h3>
        <div className="button-group">
          <button 
            className="btn btn-start"
            onClick={onStart}
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
        <h3>📊 Pipeline</h3>
        <div className="pipeline-info">
          <div className="pipeline-stage">
            <div className="stage-number">13</div>
            <div className="stage-name">Quantum Circuit Optimization</div>
          </div>
          <div className="pipeline-arrow">→</div>
          <div className="pipeline-stage">
            <div className="stage-number">14</div>
            <div className="stage-name">Photonic Integration</div>
          </div>
          <div className="pipeline-arrow">→</div>
          <div className="pipeline-stage">
            <div className="stage-number">15</div>
            <div className="stage-name">AGI Synthesis</div>
          </div>
        </div>
      </div>

      <div className="control-section">
        <h3>📋 Quick Actions</h3>
        <div className="action-buttons">
          <button className="btn btn-secondary">📈 Export Metrics</button>
          <button className="btn btn-secondary">💾 Save Config</button>
          <button className="btn btn-secondary">📋 View Logs</button>
          <button className="btn btn-secondary">🔄 Reset</button>
        </div>
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

