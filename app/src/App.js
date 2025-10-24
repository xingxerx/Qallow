import React, { useState, useEffect } from 'react';
import './App.css';
import Dashboard from './components/Dashboard';
import Terminal from './components/Terminal';
import Metrics from './components/Metrics';
import AuditLog from './components/AuditLog';
import ControlPanel from './components/ControlPanel';

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [isRunning, setIsRunning] = useState(false);
  const [metrics, setMetrics] = useState(null);
  const [terminalOutput, setTerminalOutput] = useState([]);

  useEffect(() => {
    if (window.electron) {
      window.electron.onQallowOutput((data) => {
        setTerminalOutput(prev => [...prev, { type: 'output', data, timestamp: new Date() }]);
      });

      window.electron.onQallowError((data) => {
        setTerminalOutput(prev => [...prev, { type: 'error', data, timestamp: new Date() }]);
      });
    }
  }, []);

  const handleStart = async () => {
    try {
      await window.electron.startQallow();
      setIsRunning(true);
    } catch (error) {
      console.error('Failed to start Qallow:', error);
    }
  };

  const handleStop = async () => {
    try {
      await window.electron.stopQallow();
      setIsRunning(false);
    } catch (error) {
      console.error('Failed to stop Qallow:', error);
    }
  };

  const handleRefreshMetrics = async () => {
    try {
      const data = await window.electron.getMetrics();
      setMetrics(data);
    } catch (error) {
      console.error('Failed to fetch metrics:', error);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <h1>🚀 Qallow Unified VM</h1>
          <p>Quantum-Photonic AGI System</p>
        </div>
        <div className="header-status">
          <span className={`status-badge ${isRunning ? 'running' : 'stopped'}`}>
            {isRunning ? '● Running' : '○ Stopped'}
          </span>
        </div>
      </header>

      <div className="app-container">
        <nav className="sidebar">
          <button
            className={`nav-item ${activeTab === 'dashboard' ? 'active' : ''}`}
            onClick={() => setActiveTab('dashboard')}
          >
            📊 Dashboard
          </button>
          <button
            className={`nav-item ${activeTab === 'metrics' ? 'active' : ''}`}
            onClick={() => setActiveTab('metrics')}
          >
            📈 Metrics
          </button>
          <button
            className={`nav-item ${activeTab === 'terminal' ? 'active' : ''}`}
            onClick={() => setActiveTab('terminal')}
          >
            💻 Terminal
          </button>
          <button
            className={`nav-item ${activeTab === 'audit' ? 'active' : ''}`}
            onClick={() => setActiveTab('audit')}
          >
            🔍 Audit Log
          </button>
          <button
            className={`nav-item ${activeTab === 'control' ? 'active' : ''}`}
            onClick={() => setActiveTab('control')}
          >
            ⚙️ Control
          </button>
        </nav>

        <main className="main-content">
          {activeTab === 'dashboard' && <Dashboard isRunning={isRunning} />}
          {activeTab === 'metrics' && <Metrics onRefresh={handleRefreshMetrics} />}
          {activeTab === 'terminal' && <Terminal output={terminalOutput} />}
          {activeTab === 'audit' && <AuditLog />}
          {activeTab === 'control' && (
            <ControlPanel
              isRunning={isRunning}
              onStart={handleStart}
              onStop={handleStop}
            />
          )}
        </main>
      </div>
    </div>
  );
}

export default App;

