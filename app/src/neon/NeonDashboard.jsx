import React, { useCallback, useEffect, useMemo, useState } from 'react';
import './neon.css';
import MatrixCanvas from './MatrixCanvas';

const DEFAULT_API = process.env.REACT_APP_API_URL || (typeof window !== 'undefined' ? `${window.location.protocol}//${window.location.hostname}:3001` : 'http://localhost:3001');

function useApi(base = DEFAULT_API) {
  const jget = useCallback(async (path) => {
    const res = await fetch(`${base}${path}`);
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.json();
  }, [base]);

  const jpost = useCallback(async (path, body) => {
    const res = await fetch(`${base}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {})
    });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.json();
  }, [base]);

  return { jget, jpost, base };
}

export default function NeonDashboard({ apiBase }) {
  const base = apiBase || DEFAULT_API;
  const { jget, jpost } = useApi(base);
  const [matrixOn, setMatrixOn] = useState(true);
  const [statusJson, setStatusJson] = useState('(none)');
  const [statusTs, setStatusTs] = useState('n/a');
  const [terminalText, setTerminalText] = useState('(none)');
  const [auditText, setAuditText] = useState('(none)');
  const [pending, setPending] = useState(false);

  const fmt = useCallback((obj) => JSON.stringify(obj, null, 2), []);

  const refreshStatus = useCallback(async () => {
    try {
      const st = await jget('/api/status');
      setStatusTs(new Date().toLocaleTimeString());
      setStatusJson(fmt(st));
      if (st.terminal_output && st.terminal_output.length) {
        const t = st.terminal_output.map(l => `[${l.type}] ${l.timestamp}  ${l.content}`).join('\n');
        setTerminalText(t);
      }
      if (st.audit_logs && st.audit_logs.length) {
        const a = st.audit_logs.map(l => `[${l.level}] ${l.timestamp} ${l.component} - ${l.message}`).join('\n');
        setAuditText(a);
      }
    } catch (e) {
      setStatusJson(`Error: ${e.message}`);
    }
  }, [jget, fmt]);

  const call = useCallback(async (fn) => {
    if (pending) return;
    setPending(true);
    try {
      await fn();
    } finally {
      setPending(false);
    }
  }, [pending]);

  useEffect(() => {
    refreshStatus();
    const id = setInterval(refreshStatus, 3000);
    return () => clearInterval(id);
  }, [refreshStatus]);

  // Optional WebSocket (ack echo)
  useEffect(() => {
    try {
      const wsProto = window.location.protocol === 'https:' ? 'wss' : 'ws';
      const wsUrl = `${wsProto}://${new URL(base).host}`; // point to API host
      const ws = new WebSocket(wsUrl);
      ws.onopen = () => void 0;
      ws.onmessage = () => void 0;
      ws.onerror = () => void 0;
      return () => ws.close();
    } catch {
      // ignore
    }
  }, [base]);

  return (
    <div className="neon-app">
      <MatrixCanvas enabled={matrixOn} />
      <main className="container">
        <header className="header">
          <h1>Qallow Simple Control Panel</h1>
          <div className="controls-right">
            <label className="toggle">
              <input type="checkbox" checked={matrixOn} onChange={e => setMatrixOn(e.target.checked)} />
              <span>Matrix mode</span>
            </label>
          </div>
        </header>

        <section className="row">
          <button className="btn" disabled={pending} onClick={() => call(async () => setStatusJson(fmt(await jget('/api/health'))))}>Health</button>
          <button className="btn btn-primary" disabled={pending} onClick={() => call(async () => { setStatusJson(fmt(await jpost('/api/vm/start', { ticks: 120, build: 'CPU' }))); await refreshStatus(); })}>Start VM</button>
          <button className="btn btn-danger" disabled={pending} onClick={() => call(async () => { setStatusJson(fmt(await jpost('/api/vm/stop', {}))); await refreshStatus(); })}>Stop VM</button>
          <button className="btn" disabled={pending} onClick={() => call(async () => setStatusJson(fmt(await jget('/api/metrics'))))}>Metrics</button>
          <button className="btn" disabled={pending} onClick={() => call(async () => setStatusJson(fmt(await jget('/api/logs'))))}>Logs</button>
        </section>

        <section className="row">
          <button className="btn" disabled={pending} onClick={() => call(async () => setStatusJson(fmt(await jpost('/api/quantum/run-grover', { num_qubits: 3, target_state: 5 }))))}>Run Grover</button>
          <button className="btn" disabled={pending} onClick={() => call(async () => setStatusJson(fmt(await jpost('/api/quantum/run-bell-state', {}))))}>Run Bell</button>
          <button className="btn" disabled={pending} onClick={() => call(async () => setStatusJson(fmt(await jpost('/api/quantum/run-deutsch', {}))))}>Run Deutsch</button>
          <button className="btn" disabled={pending} onClick={() => call(async () => setStatusJson(fmt(await jpost('/api/quantum/run-all', {}))))}>Run All</button>
        </section>

        <section className="panel">
          <div className="panel-title">Status <small>{statusTs}</small></div>
          <pre className="code">{statusJson}</pre>
        </section>
        <section className="panel">
          <div className="panel-title">Terminal</div>
          <pre className="code">{terminalText}</pre>
        </section>
        <section className="panel">
          <div className="panel-title">Audit Log</div>
          <pre className="code">{auditText}</pre>
        </section>
      </main>
    </div>
  );
}
