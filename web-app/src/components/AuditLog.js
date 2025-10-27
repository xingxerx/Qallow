import React, { useState } from 'react';
import './AuditLog.css';

function AuditLog({ logs }) {
  const [filter, setFilter] = useState('ALL');

  const filteredLogs = logs.filter(log => {
    if (filter === 'ALL') return true;
    return log.level === filter;
  });

  const getLevelIcon = (level) => {
    switch (level) {
      case 'Info':
        return 'ℹ️';
      case 'Success':
        return '✓';
      case 'Warning':
        return '⚠️';
      case 'Error':
        return '❌';
      default:
        return '•';
    }
  };

  return (
    <div className="audit-log-container">
      <div className="audit-header">
        <h3>🔍 Audit Log</h3>
        <div className="filter-controls">
          <select 
            value={filter} 
            onChange={(e) => setFilter(e.target.value)}
            className="filter-select"
          >
            <option>ALL</option>
            <option>Info</option>
            <option>Success</option>
            <option>Warning</option>
            <option>Error</option>
          </select>
          <span className="log-count">{filteredLogs.length} entries</span>
        </div>
      </div>

      <div className="audit-logs">
        {filteredLogs.length === 0 ? (
          <div className="audit-empty">
            No audit logs yet.
          </div>
        ) : (
          filteredLogs.map((log, idx) => (
            <div key={idx} className={`audit-entry ${log.level?.toLowerCase() || 'info'}`}>
              <div className="entry-icon">{getLevelIcon(log.level)}</div>
              <div className="entry-content">
                <div className="entry-header">
                  <span className="entry-component">{log.component}</span>
                  <span className="entry-time">
                    {new Date(log.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                <div className="entry-message">{log.message}</div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default AuditLog;

