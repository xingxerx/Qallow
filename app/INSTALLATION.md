# 🚀 Qallow Unified App - Installation Guide

Complete step-by-step installation and setup guide for the Qallow Unified App.

## System Requirements

### Minimum Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS 10.13+, or Windows 10+
- **Node.js**: 16.0.0 or higher
- **npm**: 7.0.0 or higher
- **RAM**: 4 GB
- **Disk Space**: 2 GB

### Recommended Requirements
- **OS**: Ubuntu 22.04 LTS or later
- **Node.js**: 18.0.0 or higher
- **npm**: 9.0.0 or higher
- **RAM**: 8 GB
- **Disk Space**: 5 GB
- **GPU**: NVIDIA with CUDA support (for Qallow VM)

## Pre-Installation Checklist

Before installing, ensure:

```bash
# Check Node.js version
node --version
# Should be v16.0.0 or higher

# Check npm version
npm --version
# Should be 7.0.0 or higher

# Check Qallow build exists
ls -la /root/Qallow/build/qallow_unified
# Should exist and be executable

# Check CUDA (optional but recommended)
nvidia-smi
# Should show GPU info
```

## Installation Steps

### Step 1: Navigate to App Directory

```bash
cd /root/Qallow/app
```

### Step 2: Install Node.js (if not installed)

**Ubuntu/Debian:**
```bash
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
```

**macOS:**
```bash
brew install node
```

**Windows:**
Download from https://nodejs.org/

### Step 3: Install Dependencies

```bash
npm install
```

This will install:
- React 18.2.0
- Electron 27.0.0
- Electron Builder
- React Scripts
- All other dependencies

**Installation time**: 3-5 minutes (depending on internet speed)

### Step 4: Verify Installation

```bash
# Check if node_modules exists
ls -la node_modules | head -20

# Check if dependencies are installed
npm list --depth=0
```

## Running the App

### Development Mode (with hot reload)

```bash
npm run electron-dev
```

This will:
1. Start React dev server on http://localhost:3000
2. Launch Electron app
3. Enable hot reload on file changes
4. Open developer tools

**First run**: 30-60 seconds (building React app)
**Subsequent runs**: 10-15 seconds

### Production Build

```bash
npm run electron-build
```

This will:
1. Build React app for production
2. Package with Electron Builder
3. Create installers in `dist/` directory

**Build time**: 2-5 minutes

## Troubleshooting Installation

### Issue: npm install fails

**Solution 1: Clear npm cache**
```bash
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

**Solution 2: Use npm ci instead**
```bash
npm ci
```

**Solution 3: Check Node version**
```bash
node --version
# Must be 16.0.0 or higher
```

### Issue: Electron won't start

**Solution 1: Check if display is available**
```bash
echo $DISPLAY
# Should show something like :0 or :1
```

**Solution 2: Use Xvfb for headless systems**
```bash
xvfb-run npm run electron-dev
```

**Solution 3: Check permissions**
```bash
chmod +x /root/Qallow/build/qallow_unified
```

### Issue: Port 3000 already in use**

**Solution 1: Kill process on port 3000**
```bash
lsof -ti:3000 | xargs kill -9
```

**Solution 2: Use different port**
```bash
PORT=3001 npm run electron-dev
```

### Issue: Qallow VM won't start

**Solution 1: Check binary exists**
```bash
ls -la /root/Qallow/build/qallow_unified
chmod +x /root/Qallow/build/qallow_unified
```

**Solution 2: Check CUDA**
```bash
nvidia-smi
# Should show GPU info
```

**Solution 3: Check web dashboard**
```bash
cd /root/Qallow/ui
python3 dashboard.py
# Should start on port 5000
```

## Post-Installation Setup

### 1. Start Web Dashboard (in separate terminal)

```bash
cd /root/Qallow/ui
python3 dashboard.py
```

This provides the backend API for metrics.

### 2. Start the App

```bash
cd /root/Qallow/app
npm run electron-dev
```

### 3. Verify All Components

- **Dashboard**: Should show real-time metrics
- **Terminal**: Should be ready for output
- **Metrics**: Should auto-refresh
- **Audit Log**: Should show sample events
- **Control Panel**: Should have Start/Stop buttons

## Configuration

### Environment Variables

```bash
# Set custom port for web dashboard
export PORT=5001

# Enable debug logging
export DEBUG=qallow:*

# Set custom data directory
export QALLOW_DATA_DIR=/custom/path
```

### Electron Configuration

Edit `main.js` to customize:
- Window size: `width: 1600, height: 1000`
- Dev tools: `mainWindow.webContents.openDevTools()`
- Auto-reload: `isDev` flag

## Verification Checklist

After installation, verify:

- [ ] Node.js installed: `node --version`
- [ ] npm installed: `npm --version`
- [ ] Dependencies installed: `npm list --depth=0`
- [ ] Qallow binary exists: `ls /root/Qallow/build/qallow_unified`
- [ ] App starts: `npm run electron-dev`
- [ ] Dashboard loads
- [ ] All tabs accessible
- [ ] Control Panel buttons work

## Next Steps

1. **Read QUICKSTART.md** - 5-minute quick start guide
2. **Read README.md** - Complete documentation
3. **Explore the UI** - Familiarize with all tabs
4. **Run Qallow VM** - Use Control Panel to start
5. **Monitor Execution** - Watch Terminal and Metrics

## Getting Help

### Documentation
- `/root/Qallow/app/README.md` - Full documentation
- `/root/Qallow/app/QUICKSTART.md` - Quick start guide
- `/root/Qallow/docs/` - Qallow documentation

### Logs
- App logs: Check browser console (F12)
- Qallow logs: Check Terminal tab
- System logs: `/root/Qallow/data/logs/`

### Common Issues
- See QUICKSTART.md troubleshooting section
- Check `/root/Qallow/ui/WEB_DASHBOARD_README.md`
- Review Qallow documentation

## Uninstallation

To remove the app:

```bash
# Remove node_modules
rm -rf /root/Qallow/app/node_modules

# Remove package-lock.json
rm /root/Qallow/app/package-lock.json

# Remove build artifacts
rm -rf /root/Qallow/app/build
rm -rf /root/Qallow/app/dist

# Remove entire app directory (if desired)
rm -rf /root/Qallow/app
```

## Support

For issues:
1. Check this installation guide
2. Review QUICKSTART.md
3. Check README.md
4. Review Qallow documentation
5. Check system logs

---

**Installation Complete! 🎉**

Next: Run `npm run electron-dev` to start the app!

