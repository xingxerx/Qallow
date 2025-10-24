import React, { useEffect, useRef } from 'react';
import './Terminal.css';

function Terminal({ output }) {
  const terminalRef = useRef(null);

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [output]);

  return (
    <div className="terminal-container">
      <div className="terminal-header">
        <h2>💻 Terminal Output</h2>
        <div className="terminal-info">
          <span className="output-count">{output.length} lines</span>
        </div>
      </div>

      <div className="terminal" ref={terminalRef}>
        {output.length === 0 ? (
          <div className="terminal-empty">
            <p>Waiting for output...</p>
            <p className="hint">Start the Qallow VM to see terminal output</p>
          </div>
        ) : (
          output.map((line, index) => (
            <div
              key={index}
              className={`terminal-line ${line.type}`}
            >
              <span className="timestamp">
                {line.timestamp.toLocaleTimeString()}
              </span>
              <span className="content">{line.data}</span>
            </div>
          ))
        )}
      </div>

      <div className="terminal-footer">
        <p>Real-time terminal output from Qallow Unified VM</p>
      </div>
    </div>
  );
}

export default Terminal;

