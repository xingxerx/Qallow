(function(){
  const $ = (id) => document.getElementById(id);
  const set = (id, val) => $(id).textContent = val;

  async function jget(path) {
    const r = await fetch(path);
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    return r.json();
  }
  async function jpost(path, body) {
    const r = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) });
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    return r.json();
  }

  function show(obj) { return JSON.stringify(obj, null, 2); }

  // Track last update to prevent duplicate renders
  let lastStatusUpdate = 0;
  const UPDATE_THROTTLE = 500; // ms

  async function refreshStatus() {
    const now = Date.now();
    if (now - lastStatusUpdate < UPDATE_THROTTLE) return;
    lastStatusUpdate = now;

    try {
      const st = await jget('/api/status');
      set('statusTs', new Date().toLocaleTimeString());
      set('statusOut', show(st));

      if (st.terminal_output && st.terminal_output.length) {
        const terminalText = st.terminal_output.map(l => `[${l.type}] ${l.timestamp}  ${l.content}`).join('\n');
        if ($(terminalOut).textContent !== terminalText) {
          set('terminalOut', terminalText);
        }
      }
      if (st.audit_logs && st.audit_logs.length) {
        const auditText = st.audit_logs.map(l => `[${l.level}] ${l.timestamp} ${l.component} - ${l.message}`).join('\n');
        if ($(auditOut).textContent !== auditText) {
          set('auditOut', auditText);
        }
      }
    } catch (e) {
      set('statusOut', `Error: ${e.message}`);
    }
  }

  // Track pending requests to prevent duplicate submissions
  let pendingRequest = false;

  const handleButtonClick = async (handler) => {
    if (pendingRequest) return;
    pendingRequest = true;
    try {
      await handler();
    } finally {
      pendingRequest = false;
    }
  };

  // Wire buttons (prevent duplicate handlers)
  if (!$('btnHealth').onclick) {
    $('btnHealth').onclick = () => handleButtonClick(async () => {
      try { set('statusOut', show(await jget('/api/health'))); } catch (e) { set('statusOut', e.message); }
    });
  }
  if (!$('btnStart').onclick) {
    $('btnStart').onclick = () => handleButtonClick(async () => {
      try { set('statusOut', show(await jpost('/api/vm/start', { ticks: 120, build: 'CPU' }))); await refreshStatus(); } catch (e) { set('statusOut', e.message); }
    });
  }
  if (!$('btnStop').onclick) {
    $('btnStop').onclick = () => handleButtonClick(async () => {
      try { set('statusOut', show(await jpost('/api/vm/stop', {}))); await refreshStatus(); } catch (e) { set('statusOut', e.message); }
    });
  }
  if (!$('btnMetrics').onclick) {
    $('btnMetrics').onclick = () => handleButtonClick(async () => {
      try { set('statusOut', show(await jget('/api/metrics'))); } catch (e) { set('statusOut', e.message); }
    });
  }
  if (!$('btnLogs').onclick) {
    $('btnLogs').onclick = () => handleButtonClick(async () => {
      try { set('statusOut', show(await jget('/api/logs'))); } catch (e) { set('statusOut', e.message); }
    });
  }
  if (!$('btnGrover').onclick) {
    $('btnGrover').onclick = () => handleButtonClick(async () => {
      try { set('statusOut', show(await jpost('/api/quantum/run-grover', { num_qubits: 3, target_state: 5 }))); } catch (e) { set('statusOut', e.message); }
    });
  }
  if (!$('btnBell').onclick) {
    $('btnBell').onclick = () => handleButtonClick(async () => {
      try { set('statusOut', show(await jpost('/api/quantum/run-bell-state', {}))); } catch (e) { set('statusOut', e.message); }
    });
  }
  if (!$('btnDeutsch').onclick) {
    $('btnDeutsch').onclick = () => handleButtonClick(async () => {
      try { set('statusOut', show(await jpost('/api/quantum/run-deutsch', {}))); } catch (e) { set('statusOut', e.message); }
    });
  }
  if (!$('btnAll').onclick) {
    $('btnAll').onclick = () => handleButtonClick(async () => {
      try { set('statusOut', show(await jpost('/api/quantum/run-all', {}))); } catch (e) { set('statusOut', e.message); }
    });
  }

  // Optional WebSocket (for future real-time stream)
  try {
    const wsProto = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${wsProto}://${location.host}`);
    ws.onopen = () => console.log('[ws] connected');
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if(msg && msg.type === 'ack'){
          // Could surface to UI if needed
        }
      } catch {}
    };
    ws.onerror = () => console.log('[ws] error');
    ws.onclose = () => console.log('[ws] closed');
  } catch {}

  refreshStatus();
  setInterval(refreshStatus, 3000);
})();
