import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import Dashboard from './components/Dashboard';
import Terminal from './components/Terminal';
import Metrics from './components/Metrics';
import AuditLog from './components/AuditLog';
import ControlPanel from './components/ControlPanel';

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [vmRunning, setVmRunning] = useState(false);
  const [terminalOutput, setTerminalOutput] = useState([]);
  const [metrics, setMetrics] = useState({});
  const [auditLogs, setAuditLogs] = useState([]);
  const [loading, setLoading] = useState(false);

  const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:3001/api';

  useEffect(() => {
    // Poll for status
    const interval = setInterval(async () => {
      try {
        const response = await axios.get(`${API_BASE}/status`);
        setVmRunning(response.data.vm_running);
        if (response.data.terminal_output) {
          setTerminalOutput(response.data.terminal_output);
        }
        if (response.data.metrics) {
          setMetrics(response.data.metrics);
        }
        if (response.data.audit_logs) {
          setAuditLogs(response.data.audit_logs);
        }
      } catch (error) {
        console.error('Failed to fetch status:', error);
      }
    }, 1000);

    return () => clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleStartVM = async () => {
    setLoading(true);
    try {
      await axios.post(`${API_BASE}/vm/start`);
      setVmRunning(true);
    } catch (error) {
      console.error('Failed to start VM:', error);
      alert('Error starting VM: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleStopVM = async () => {
    setLoading(true);
    try {
      await axios.post(`${API_BASE}/vm/stop`);
      setVmRunning(false);
    } catch (error) {
      console.error('Failed to stop VM:', error);
      alert('Error stopping VM: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>🚀 Qallow Unified System</h1>
        <div className="status-indicator">
          <span className={`status ${vmRunning ? 'running' : 'stopped'}`}>
            {vmRunning ? '● Running' : '● Stopped'}
          </span>
        </div>
      </header>

      <nav className="app-nav">
        <button 
          className={`nav-btn ${activeTab === 'dashboard' ? 'active' : ''}`}
          onClick={() => setActiveTab('dashboard')}
        >
          📊 Dashboard
        </button>
        <button 
          className={`nav-btn ${activeTab === 'terminal' ? 'active' : ''}`}
          onClick={() => setActiveTab('terminal')}
        >
          💻 Terminal
        </button>
        <button 
          className={`nav-btn ${activeTab === 'metrics' ? 'active' : ''}`}
          onClick={() => setActiveTab('metrics')}
        >
          📈 Metrics
        </button>
        <button 
          className={`nav-btn ${activeTab === 'audit' ? 'active' : ''}`}
          onClick={() => setActiveTab('audit')}
        >
          🔍 Audit Log
        </button>
        <button 
          className={`nav-btn ${activeTab === 'control' ? 'active' : ''}`}
          onClick={() => setActiveTab('control')}
        >
          ⚙️ Control
        </button>
      </nav>

      <main className="app-main">
        {activeTab === 'dashboard' && <Dashboard metrics={metrics} vmRunning={vmRunning} />}
        {activeTab === 'terminal' && <Terminal output={terminalOutput} />}
        {activeTab === 'metrics' && <Metrics data={metrics} />}
        {activeTab === 'audit' && <AuditLog logs={auditLogs} />}
        {activeTab === 'control' && (
          <ControlPanel 
            vmRunning={vmRunning}
            onStart={handleStartVM}
            onStop={handleStopVM}
            loading={loading}
          />
        )}
      </main>
    </div>
  );
}

export default App;

