const { app, BrowserWindow, Menu, ipcMain } = require('electron');
const path = require('path');
const isDev = require('electron-is-dev');
const { spawn } = require('child_process');

let mainWindow;
let qallowProcess;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1600,
    height: 1000,
    minWidth: 1200,
    minHeight: 800,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true,
      enableRemoteModule: false
    },
    icon: path.join(__dirname, 'assets/icon.png')
  });

  const startUrl = isDev
    ? 'http://localhost:3000'
    : `file://${path.join(__dirname, '../build/index.html')}`;

  mainWindow.loadURL(startUrl);

  if (isDev) {
    mainWindow.webContents.openDevTools();
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
    if (qallowProcess) {
      qallowProcess.kill();
    }
  });
}

app.on('ready', createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (mainWindow === null) {
    createWindow();
  }
});

// IPC Handlers
ipcMain.handle('start-qallow', async () => {
  return new Promise((resolve, reject) => {
    try {
      qallowProcess = spawn('./build/qallow_unified', [], {
        cwd: path.join(__dirname, '..')
      });

      qallowProcess.stdout.on('data', (data) => {
        mainWindow?.webContents.send('qallow-output', data.toString());
      });

      qallowProcess.stderr.on('data', (data) => {
        mainWindow?.webContents.send('qallow-error', data.toString());
      });

      resolve({ status: 'started' });
    } catch (error) {
      reject(error);
    }
  });
});

ipcMain.handle('stop-qallow', async () => {
  if (qallowProcess) {
    qallowProcess.kill();
    qallowProcess = null;
    return { status: 'stopped' };
  }
  return { status: 'not running' };
});

ipcMain.handle('get-metrics', async () => {
  try {
    const response = await fetch('http://localhost:5000/api/phases');
    return await response.json();
  } catch (error) {
    return { error: error.message };
  }
});

ipcMain.handle('get-audit-logs', async () => {
  try {
    const response = await fetch('http://localhost:5000/api/audit');
    return await response.json();
  } catch (error) {
    return { error: error.message };
  }
});

