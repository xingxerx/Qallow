import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './ServerDashboard.css';

const ServerDashboard = () => {
  const [serverStatus, setServerStatus] = useState('connecting');
  const [quantumStatus, setQuantumStatus] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [ws, setWs] = useState(null);

  const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

  // Initialize WebSocket connection
  useEffect(() => {
    try {
      const wsUrl = `ws://${window.location.hostname}:5000`;
      const websocket = new WebSocket(wsUrl);

      websocket.onopen = () => {
        console.log('WebSocket connected');
        setServerStatus('connected');
      };

      websocket.onmessage = (event) => {
        console.log('WebSocket message:', event.data);
      };

      websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        setServerStatus('error');
      };

      websocket.onclose = () => {
        console.log('WebSocket disconnected');
        setServerStatus('disconnected');
      };

      setWs(websocket);

      return () => {
        if (websocket) websocket.close();
      };
    } catch (err) {
      console.error('WebSocket setup error:', err);
      setServerStatus('error');
    }
  }, []);

  // Check server health
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await axios.get(`${API_URL}/health`);
        if (response.data.success) {
          setServerStatus('healthy');
        }
      } catch (err) {
        console.error('Health check failed:', err);
        setServerStatus('unhealthy');
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 5000);
    return () => clearInterval(interval);
  }, []);

  // Get quantum status
  useEffect(() => {
    const getQuantumStatus = async () => {
      try {
        const response = await axios.get(`${API_URL}/quantum/status`);
        setQuantumStatus(response.data.data);
      } catch (err) {
        console.error('Failed to get quantum status:', err);
        setError(err.message);
      }
    };

    getQuantumStatus();
    const interval = setInterval(getQuantumStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  // Get system metrics
  useEffect(() => {
    const getMetrics = async () => {
      try {
        const response = await axios.get(`${API_URL}/system/metrics`);
        setMetrics(response.data.data);
      } catch (err) {
        console.error('Failed to get metrics:', err);
      }
    };

    getMetrics();
    const interval = setInterval(getMetrics, 5000);
    return () => clearInterval(interval);
  }, []);

  // Run quantum algorithms
  const runAlgorithm = async (algorithm) => {
    setLoading(true);
    setError(null);
    try {
      let endpoint = '';
      let payload = {};

      switch (algorithm) {
        case 'grover':
          endpoint = '/quantum/run-grover';
          payload = { num_qubits: 3, target_state: 5 };
          break;
        case 'bell':
          endpoint = '/quantum/run-bell-state';
          break;
        case 'deutsch':
          endpoint = '/quantum/run-deutsch';
          break;
        case 'all':
          endpoint = '/quantum/run-all';
          break;
        default:
          throw new Error('Unknown algorithm');
      }

      const response = await axios.post(`${API_URL}${endpoint}`, payload);
      setResults(response.data.data);
    } catch (err) {
      console.error(`Failed to run ${algorithm}:`, err);
      setError(err.response?.data?.error || err.message);
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'healthy':
      case 'connected':
        return '#10b981';
      case 'unhealthy':
      case 'error':
        return '#ef4444';
      case 'connecting':
      case 'disconnected':
        return '#f59e0b';
      default:
        return '#6b7280';
    }
  };

  return (
    <div className="server-dashboard">
      <header className="dashboard-header">
        <h1>🚀 Qallow Server Dashboard</h1>
        <p>Unified Quantum Framework Management</p>
      </header>

      {/* Status Cards */}
      <div className="status-grid">
        <div className="status-card">
          <div className="status-indicator" style={{ backgroundColor: getStatusColor(serverStatus) }}></div>
          <h3>Server Status</h3>
          <p className="status-text">{serverStatus.toUpperCase()}</p>
        </div>

        {quantumStatus && (
          <div className="status-card">
            <div className="status-indicator" style={{ backgroundColor: '#3b82f6' }}></div>
            <h3>Quantum Framework</h3>
            <p className="status-text">{quantumStatus.framework}</p>
            <p className="status-subtext">{quantumStatus.simulator}</p>
          </div>
        )}

        {metrics && (
          <div className="status-card">
            <h3>System Uptime</h3>
            <p className="status-text">{Math.floor(metrics.uptime)}s</p>
            <p className="status-subtext">
              Memory: {Math.round(metrics.memory.heapUsed / 1024 / 1024)}MB
            </p>
          </div>
        )}
      </div>

      {/* Error Display */}
      {error && (
        <div className="error-box">
          <strong>⚠️ Error:</strong> {error}
        </div>
      )}

      {/* Algorithm Controls */}
      <div className="controls-section">
        <h2>Quantum Algorithms</h2>
        <div className="button-grid">
          <button
            onClick={() => runAlgorithm('grover')}
            disabled={loading}
            className="btn btn-primary"
          >
            {loading ? '⏳ Running...' : '🔍 Grover\'s Algorithm'}
          </button>
          <button
            onClick={() => runAlgorithm('bell')}
            disabled={loading}
            className="btn btn-primary"
          >
            {loading ? '⏳ Running...' : '🔗 Bell State'}
          </button>
          <button
            onClick={() => runAlgorithm('deutsch')}
            disabled={loading}
            className="btn btn-primary"
          >
            {loading ? '⏳ Running...' : '⚙️ Deutsch Algorithm'}
          </button>
          <button
            onClick={() => runAlgorithm('all')}
            disabled={loading}
            className="btn btn-success"
          >
            {loading ? '⏳ Running...' : '▶️ Run All'}
          </button>
        </div>
      </div>

      {/* Results Display */}
      {results && (
        <div className="results-section">
          <h2>Results</h2>
          <pre className="results-box">
            {typeof results === 'string' ? results : JSON.stringify(results, null, 2)}
          </pre>
        </div>
      )}

      {/* Metrics Display */}
      {metrics && (
        <div className="metrics-section">
          <h2>System Metrics</h2>
          <div className="metrics-grid">
            <div className="metric-card">
              <h4>Heap Used</h4>
              <p>{Math.round(metrics.memory.heapUsed / 1024 / 1024)} MB</p>
            </div>
            <div className="metric-card">
              <h4>Heap Total</h4>
              <p>{Math.round(metrics.memory.heapTotal / 1024 / 1024)} MB</p>
            </div>
            <div className="metric-card">
              <h4>External</h4>
              <p>{Math.round(metrics.memory.external / 1024 / 1024)} MB</p>
            </div>
            <div className="metric-card">
              <h4>Uptime</h4>
              <p>{Math.floor(metrics.uptime)} seconds</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ServerDashboard;

