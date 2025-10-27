import React from 'react';
import './Metrics.css';

function Metrics({ data }) {
  const metrics = [
    { label: 'Fidelity', value: data.fidelity, unit: '%', format: (v) => (v * 100).toFixed(1) },
    { label: 'Energy', value: data.energy, unit: 'J', format: (v) => v.toFixed(3) },
    { label: 'Risk', value: data.risk, unit: '', format: (v) => v.toFixed(3) },
    { label: 'Reward', value: data.reward, unit: '', format: (v) => v.toFixed(3) },
    { label: 'Coherence', value: data.coherence, unit: '', format: (v) => v.toFixed(3) },
    { label: 'Entanglement', value: data.entanglement, unit: '', format: (v) => v.toFixed(3) },
  ];

  return (
    <div className="metrics-container">
      <h3>📈 System Metrics</h3>
      <div className="metrics-grid">
        {metrics.map((metric, idx) => (
          <div key={idx} className="metric-card">
            <div className="metric-label">{metric.label}</div>
            <div className="metric-value">
              {metric.value !== undefined ? metric.format(metric.value) : 'N/A'}
              <span className="metric-unit">{metric.unit}</span>
            </div>
            <div className="metric-bar">
              <div 
                className="metric-fill"
                style={{ width: `${Math.min((metric.value || 0) * 100, 100)}%` }}
              />
            </div>
          </div>
        ))}
      </div>

      <div className="metrics-details">
        <h4>Detailed Metrics</h4>
        <div className="details-table">
          <div className="table-row header">
            <div className="table-cell">Metric</div>
            <div className="table-cell">Value</div>
            <div className="table-cell">Status</div>
          </div>
          {metrics.map((metric, idx) => (
            <div key={idx} className="table-row">
              <div className="table-cell">{metric.label}</div>
              <div className="table-cell">
                {metric.value !== undefined ? metric.format(metric.value) : 'N/A'} {metric.unit}
              </div>
              <div className="table-cell">
                <span className={`status-badge ${metric.value > 0.7 ? 'good' : 'warning'}`}>
                  {metric.value > 0.7 ? '✓ Good' : '⚠ Monitor'}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default Metrics;

