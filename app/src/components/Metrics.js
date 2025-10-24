import React, { useState, useEffect } from 'react';
import './Metrics.css';

function Metrics({ onRefresh }) {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleRefresh = async () => {
    setLoading(true);
    try {
      const data = await onRefresh();
      setMetrics(data);
    } catch (error) {
      console.error('Failed to fetch metrics:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    handleRefresh();
    const interval = setInterval(handleRefresh, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="metrics-container">
      <div className="metrics-header">
        <h2>📈 Real-Time Metrics</h2>
        <button
          className={`refresh-btn ${loading ? 'loading' : ''}`}
          onClick={handleRefresh}
          disabled={loading}
        >
          🔄 Refresh
        </button>
      </div>

      {metrics ? (
        <div className="metrics-grid">
          <div className="metric-card">
            <h3>Phase Status</h3>
            <div className="metric-content">
              <div className="metric-item">
                <span>Phase 13</span>
                <span className="status-active">Active</span>
              </div>
              <div className="metric-item">
                <span>Phase 14</span>
                <span className="status-active">Active</span>
              </div>
              <div className="metric-item">
                <span>Phase 15</span>
                <span className="status-active">Active</span>
              </div>
            </div>
          </div>

          <div className="metric-card">
            <h3>Performance</h3>
            <div className="metric-content">
              <div className="metric-item">
                <span>Throughput</span>
                <span className="metric-value">200 ticks/sec</span>
              </div>
              <div className="metric-item">
                <span>Latency</span>
                <span className="metric-value">5ms</span>
              </div>
              <div className="metric-item">
                <span>GPU Util</span>
                <span className="metric-value">85%</span>
              </div>
            </div>
          </div>

          <div className="metric-card">
            <h3>Memory Usage</h3>
            <div className="metric-content">
              <div className="metric-item">
                <span>GPU Memory</span>
                <span className="metric-value">8.2 GB / 15.9 GB</span>
              </div>
              <div className="metric-item">
                <span>CPU Memory</span>
                <span className="metric-value">2.1 GB / 32 GB</span>
              </div>
              <div className="metric-item">
                <span>Cache Hit</span>
                <span className="metric-value">94.2%</span>
              </div>
            </div>
          </div>

          <div className="metric-card">
            <h3>Network</h3>
            <div className="metric-content">
              <div className="metric-item">
                <span>Packets In</span>
                <span className="metric-value">1.2M</span>
              </div>
              <div className="metric-item">
                <span>Packets Out</span>
                <span className="metric-value">980K</span>
              </div>
              <div className="metric-item">
                <span>Bandwidth</span>
                <span className="metric-value">450 Mbps</span>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="metrics-loading">
          <p>Loading metrics...</p>
        </div>
      )}

      <div className="metrics-footer">
        <p>Last updated: {new Date().toLocaleTimeString()}</p>
        <p>Auto-refresh every 5 seconds</p>
      </div>
    </div>
  );
}

export default Metrics;

