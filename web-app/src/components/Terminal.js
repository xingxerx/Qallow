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
        <h3>💻 Live Terminal Output</h3>
        <div className="terminal-info">
          {output.length} lines
        </div>
      </div>
      <div className="terminal" ref={terminalRef}>
        {output.length === 0 ? (
          <div className="terminal-empty">
            No output yet. Start the VM to see logs.
          </div>
        ) : (
          output.map((line, idx) => (
            <div key={idx} className={`terminal-line ${line.type || 'info'}`}>
              <span className="terminal-timestamp">
                [{new Date(line.timestamp).toLocaleTimeString()}]
              </span>
              <span className="terminal-content">{line.content}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default Terminal;

