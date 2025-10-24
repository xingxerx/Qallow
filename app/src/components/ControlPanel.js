import React, { useState } from 'react';
import './ControlPanel.css';

function ControlPanel({ isRunning, onStart, onStop }) {
  const [selectedPhase, setSelectedPhase] = useState('14');
  const [ticks, setTicks] = useState('1000');
  const [parameters, setParameters] = useState({
    target_fidelity: '0.981',
    eps: '5e-6'
  });

  const phases = [
    { id: '13', name: 'Phase 13: Harmonic Propagation', ticks: '100' },
    { id: '14', name: 'Phase 14: Coherence-Lattice Integration', ticks: '300' },
    { id: '15', name: 'Phase 15: Convergence & Lock-In', ticks: '400' }
  ];

  const handleParameterChange = (key, value) => {
    setParameters(prev => ({ ...prev, [key]: value }));
  };

  return (
    <div className="control-panel">
      <div className="control-section">
        <h2>🎮 System Control</h2>
        
        <div className="control-buttons">
          <button
            className={`control-btn start-btn ${isRunning ? 'disabled' : ''}`}
            onClick={onStart}
            disabled={isRunning}
          >
            ▶️ Start VM
          </button>
          <button
            className={`control-btn stop-btn ${!isRunning ? 'disabled' : ''}`}
            onClick={onStop}
            disabled={!isRunning}
          >
            ⏹️ Stop VM
          </button>
        </div>

        <div className="status-display">
          <div className="status-item">
            <span>System Status:</span>
            <span className={`status-value ${isRunning ? 'running' : 'stopped'}`}>
              {isRunning ? '🟢 Running' : '🔴 Stopped'}
            </span>
          </div>
        </div>
      </div>

      <div className="control-section">
        <h2>⚙️ Phase Configuration</h2>
        
        <div className="form-group">
          <label>Select Phase</label>
          <select
            value={selectedPhase}
            onChange={(e) => setSelectedPhase(e.target.value)}
            className="form-select"
          >
            {phases.map(phase => (
              <option key={phase.id} value={phase.id}>
                {phase.name}
              </option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <label>Number of Ticks</label>
          <input
            type="number"
            value={ticks}
            onChange={(e) => setTicks(e.target.value)}
            className="form-input"
            min="1"
            max="10000"
          />
        </div>

        <div className="form-group">
          <label>Target Fidelity</label>
          <input
            type="number"
            value={parameters.target_fidelity}
            onChange={(e) => handleParameterChange('target_fidelity', e.target.value)}
            className="form-input"
            step="0.001"
            min="0"
            max="1"
          />
        </div>

        <div className="form-group">
          <label>Epsilon (eps)</label>
          <input
            type="text"
            value={parameters.eps}
            onChange={(e) => handleParameterChange('eps', e.target.value)}
            className="form-input"
            placeholder="e.g., 5e-6"
          />
        </div>

        <button className="run-phase-btn">
          ▶️ Run Phase {selectedPhase}
        </button>
      </div>

      <div className="control-section">
        <h2>📊 Quick Actions</h2>
        
        <div className="quick-actions">
          <button className="action-btn">
            📈 Export Metrics
          </button>
          <button className="action-btn">
            💾 Save Configuration
          </button>
          <button className="action-btn">
            🔄 Reset System
          </button>
          <button className="action-btn">
            📋 View Logs
          </button>
        </div>
      </div>

      <div className="control-section">
        <h2>ℹ️ System Information</h2>
        
        <div className="info-grid">
          <div className="info-item">
            <span>Build Version:</span>
            <span>1.0.0</span>
          </div>
          <div className="info-item">
            <span>GPU:</span>
            <span>NVIDIA RTX 5080</span>
          </div>
          <div className="info-item">
            <span>CUDA Version:</span>
            <span>12.0</span>
          </div>
          <div className="info-item">
            <span>Memory:</span>
            <span>15.9 GB</span>
          </div>
          <div className="info-item">
            <span>Uptime:</span>
            <span>2h 34m</span>
          </div>
          <div className="info-item">
            <span>Last Update:</span>
            <span>{new Date().toLocaleTimeString()}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ControlPanel;

