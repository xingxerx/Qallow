import React from 'react';
import './Dashboard.css';

function Dashboard({ metrics, vmRunning }) {
  return (
    <div className="dashboard">
      <div className="dashboard-grid">
        <div className="stat-card">
          <div className="stat-label">Status</div>
          <div className={`stat-value ${vmRunning ? 'running' : 'stopped'}`}>
            {vmRunning ? '🟢 Running' : '🔴 Stopped'}
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-label">Phase</div>
          <div className="stat-value">13 → 14 → 15</div>
          <div className="stat-desc">Unified Pipeline</div>
        </div>

        <div className="stat-card">
          <div className="stat-label">Fidelity</div>
          <div className="stat-value">
            {metrics.fidelity ? (metrics.fidelity * 100).toFixed(1) : '0'}%
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-label">Energy</div>
          <div className="stat-value">
            {metrics.energy ? metrics.energy.toFixed(2) : '0.00'}
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-label">Risk</div>
          <div className="stat-value">
            {metrics.risk ? metrics.risk.toFixed(2) : '0.00'}
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-label">Reward</div>
          <div className="stat-value">
            {metrics.reward ? metrics.reward.toFixed(2) : '0.00'}
          </div>
        </div>
      </div>

      <div className="info-section">
        <h3>System Information</h3>
        <div className="info-grid">
          <div className="info-item">
            <span className="info-label">Build Type:</span>
            <span className="info-value">CPU</span>
          </div>
          <div className="info-item">
            <span className="info-label">Ticks:</span>
            <span className="info-value">1000</span>
          </div>
          <div className="info-item">
            <span className="info-label">Mode:</span>
            <span className="info-value">Unified System</span>
          </div>
          <div className="info-item">
            <span className="info-label">Quantum Backend:</span>
            <span className="info-value">CUDA-Q</span>
          </div>
        </div>
      </div>

      <div className="phases-section">
        <h3>Phase Pipeline</h3>
        <div className="phase-flow">
          <div className="phase-box">
            <div className="phase-number">13</div>
            <div className="phase-name">Quantum Circuit<br/>Optimization</div>
          </div>
          <div className="phase-arrow">→</div>
          <div className="phase-box">
            <div className="phase-number">14</div>
            <div className="phase-name">Photonic<br/>Integration</div>
          </div>
          <div className="phase-arrow">→</div>
          <div className="phase-box">
            <div className="phase-number">15</div>
            <div className="phase-name">AGI<br/>Synthesis</div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;

