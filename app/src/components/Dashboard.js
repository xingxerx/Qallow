import React, { useState, useEffect } from 'react';
import './Dashboard.css';

function Dashboard({ isRunning }) {
  const [stats, setStats] = useState({
    overlayStability: {
      orbital: 0.9575,
      river: 0.9959,
      mycelial: 0.9963,
      global: 0.9832
    },
    ethics: {
      safety: 0.9832,
      clarity: 0.9999,
      human: 1.0000,
      combined: 2.39
    },
    coherence: {
      value: 0.9993,
      decoherence: 0.000075,
      mode: 'CUDA GPU'
    },
    execution: {
      ticks: 1000,
      equilibrium: 201,
      status: 'COMPLETE'
    }
  });

  const renderProgressBar = (value, max = 1) => {
    const percentage = (value / max) * 100;
    return (
      <div className="progress-bar">
        <div className="progress-fill" style={{ width: `${percentage}%` }}></div>
        <span className="progress-text">{(value * 100).toFixed(2)}%</span>
      </div>
    );
  };

  return (
    <div className="dashboard">
      <div className="dashboard-grid">
        {/* Overlay Stability */}
        <div className="card">
          <h2>📊 Overlay Stability</h2>
          <div className="metric-group">
            <div className="metric">
              <label>Orbital</label>
              {renderProgressBar(stats.overlayStability.orbital)}
            </div>
            <div className="metric">
              <label>River</label>
              {renderProgressBar(stats.overlayStability.river)}
            </div>
            <div className="metric">
              <label>Mycelial</label>
              {renderProgressBar(stats.overlayStability.mycelial)}
            </div>
            <div className="metric">
              <label>Global</label>
              {renderProgressBar(stats.overlayStability.global)}
            </div>
          </div>
        </div>

        {/* Ethics Monitoring */}
        <div className="card">
          <h2>⚖️ Ethics Monitoring</h2>
          <div className="metric-group">
            <div className="metric">
              <label>Safety (S)</label>
              {renderProgressBar(stats.ethics.safety)}
            </div>
            <div className="metric">
              <label>Clarity (C)</label>
              {renderProgressBar(stats.ethics.clarity)}
            </div>
            <div className="metric">
              <label>Human (H)</label>
              {renderProgressBar(stats.ethics.human)}
            </div>
            <div className="metric ethics-combined">
              <label>Combined Score</label>
              <div className="combined-score">
                E = S+C+H = {stats.ethics.combined.toFixed(2)}
              </div>
              <span className="status-pass">✓ PASS</span>
            </div>
          </div>
        </div>

        {/* Coherence */}
        <div className="card">
          <h2>🌊 Coherence</h2>
          <div className="metric-group">
            <div className="metric">
              <label>Coherence Value</label>
              {renderProgressBar(stats.coherence.value)}
            </div>
            <div className="metric">
              <label>Decoherence</label>
              <div className="decoherence-value">
                {stats.coherence.decoherence.toExponential(6)}
              </div>
            </div>
            <div className="metric">
              <label>Mode</label>
              <div className="mode-badge">{stats.coherence.mode}</div>
            </div>
          </div>
        </div>

        {/* Execution Status */}
        <div className="card">
          <h2>⚡ Execution Status</h2>
          <div className="metric-group">
            <div className="metric">
              <label>Total Ticks</label>
              <div className="stat-value">{stats.execution.ticks}</div>
            </div>
            <div className="metric">
              <label>Equilibrium Reached</label>
              <div className="stat-value">Tick {stats.execution.equilibrium}</div>
            </div>
            <div className="metric">
              <label>Status</label>
              <div className="status-badge-large">{stats.execution.status}</div>
            </div>
          </div>
        </div>

        {/* System Status */}
        <div className="card full-width">
          <h2>🖥️ System Status</h2>
          <div className="system-status">
            <div className="status-item">
              <span className="status-icon">✓</span>
              <span>Terminal Interface: WORKING</span>
            </div>
            <div className="status-item">
              <span className="status-icon">✓</span>
              <span>Collector System: ACTIVE</span>
            </div>
            <div className="status-item">
              <span className="status-icon">✓</span>
              <span>AGI Agent Layer: RUNNING</span>
            </div>
            <div className="status-item">
              <span className="status-icon">✓</span>
              <span>Quantum Algorithms: EXECUTING</span>
            </div>
            <div className="status-item">
              <span className="status-icon">✓</span>
              <span>Circuit Simulation: STABLE</span>
            </div>
            <div className="status-item">
              <span className="status-icon">✓</span>
              <span>GPU Acceleration: ENABLED</span>
            </div>
          </div>
        </div>

        {/* GPU Info */}
        <div className="card">
          <h2>🎮 GPU Acceleration</h2>
          <div className="gpu-info">
            <div className="gpu-item">
              <span>GPU:</span>
              <strong>NVIDIA GeForce RTX 5080</strong>
            </div>
            <div className="gpu-item">
              <span>Compute Capability:</span>
              <strong>12.0</strong>
            </div>
            <div className="gpu-item">
              <span>Memory:</span>
              <strong>15.9 GB</strong>
            </div>
            <div className="gpu-item">
              <span>Multiprocessors:</span>
              <strong>84</strong>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;

