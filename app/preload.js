const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electron', {
  startQallow: () => ipcRenderer.invoke('start-qallow'),
  stopQallow: () => ipcRenderer.invoke('stop-qallow'),
  getMetrics: () => ipcRenderer.invoke('get-metrics'),
  getAuditLogs: () => ipcRenderer.invoke('get-audit-logs'),
  onQallowOutput: (callback) => ipcRenderer.on('qallow-output', (event, data) => callback(data)),
  onQallowError: (callback) => ipcRenderer.on('qallow-error', (event, data) => callback(data))
});

