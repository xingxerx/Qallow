import React, { useState, useEffect } from 'react';
import './AuditLog.css';

function AuditLog() {
  const [logs, setLogs] = useState([
    {
      id: 1,
      timestamp: new Date(Date.now() - 60000),
      level: 'INFO',
      component: 'VM',
      message: 'System initialized successfully'
    },
    {
      id: 2,
      timestamp: new Date(Date.now() - 50000),
      level: 'INFO',
      component: 'GPU',
      message: 'CUDA GPU acceleration enabled'
    },
    {
      id: 3,
      timestamp: new Date(Date.now() - 40000),
      level: 'INFO',
      component: 'ETHICS',
      message: 'Ethics monitoring started'
    },
    {
      id: 4,
      timestamp: new Date(Date.now() - 30000),
      level: 'SUCCESS',
      component: 'PHASE14',
      message: 'Coherence-lattice integration complete'
    },
    {
      id: 5,
      timestamp: new Date(Date.now() - 20000),
      level: 'SUCCESS',
      component: 'PHASE15',
      message: 'Convergence & lock-in achieved'
    },
    {
      id: 6,
      timestamp: new Date(Date.now() - 10000),
      level: 'SUCCESS',
      component: 'EQUILIBRIUM',
      message: 'Stable equilibrium reached at tick 201'
    }
  ]);

  const [filter, setFilter] = useState('ALL');

  const filteredLogs = filter === 'ALL' ? logs : logs.filter(log => log.level === filter);

  const getLevelColor = (level) => {
    switch (level) {
      case 'ERROR':
        return '#ff6464';
      case 'WARNING':
        return '#ffaa00';
      case 'SUCCESS':
        return '#00ff64';
      case 'INFO':
      default:
        return '#00d4ff';
    }
  };

  return (
    <div className="audit-log-container">
      <div className="audit-header">
        <h2>🔍 Audit Log</h2>
        <div className="filter-buttons">
          {['ALL', 'INFO', 'SUCCESS', 'WARNING', 'ERROR'].map(level => (
            <button
              key={level}
              className={`filter-btn ${filter === level ? 'active' : ''}`}
              onClick={() => setFilter(level)}
            >
              {level}
            </button>
          ))}
        </div>
      </div>

      <div className="audit-logs">
        {filteredLogs.length === 0 ? (
          <div className="no-logs">
            <p>No logs found for filter: {filter}</p>
          </div>
        ) : (
          filteredLogs.map(log => (
            <div key={log.id} className="log-entry">
              <div className="log-level" style={{ borderLeftColor: getLevelColor(log.level) }}>
                <span className="level-badge" style={{ backgroundColor: getLevelColor(log.level) }}>
                  {log.level}
                </span>
              </div>
              <div className="log-content">
                <div className="log-header">
                  <span className="component">{log.component}</span>
                  <span className="timestamp">
                    {log.timestamp.toLocaleTimeString()}
                  </span>
                </div>
                <div className="log-message">{log.message}</div>
              </div>
            </div>
          ))
        )}
      </div>

      <div className="audit-footer">
        <p>Total logs: {logs.length} | Filtered: {filteredLogs.length}</p>
      </div>
    </div>
  );
}

export default AuditLog;

