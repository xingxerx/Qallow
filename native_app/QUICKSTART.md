# ⚡ Qallow Native App - Quick Start Guide

Get the Qallow Native Desktop Application running in 5 minutes!

## Prerequisites

- Rust 1.70.0+ installed
- Qallow built (binaries in `/root/Qallow/build/`)
- Linux, macOS, or Windows

## Installation (2 minutes)

### Step 1: Install Rust (if not already installed)

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### Step 2: Build the Application

```bash
cd /root/Qallow/native_app
cargo build --release
```

**Build time**: 2-5 minutes (first time), 10-30 seconds (subsequent)

## Running the App (1 minute)

### Option 1: Using Cargo

```bash
cd /root/Qallow/native_app
cargo run --release
```

### Option 2: Using Compiled Binary

```bash
./target/release/qallow-native
```

The application window will open automatically!

## First Steps (2 minutes)

### 1. Explore the Dashboard

- View real-time metrics
- Check system status
- Monitor overlay stability

### 2. Check Metrics

- See performance data
- Monitor memory usage
- Review network statistics

### 3. Start the VM

1. Go to **Control Panel** tab
2. Select **CPU** or **CUDA** build
3. Select **Phase 14**
4. Click **▶️ Start VM**

### 4. Monitor Execution

- Watch **Terminal** for live output
- Check **Metrics** for performance
- Review **Audit Log** for events

## Common Tasks

### Run Phase 13 (Harmonic Propagation)

1. Control Panel → Select Phase 13
2. Set Ticks: 400
3. Click Start VM
4. Monitor in Terminal

### Run Phase 14 (Coherence-Lattice)

1. Control Panel → Select Phase 14
2. Set Ticks: 600
3. Set Target Fidelity: 0.981
4. Click Start VM

### Run Phase 15 (Convergence & Lock-In)

1. Control Panel → Select Phase 15
2. Set Ticks: 800
3. Set Epsilon: 5e-6
4. Click Start VM

### Export Metrics

1. Control Panel → Click **📈 Export Metrics**
2. Choose format (CSV, JSON)
3. Select save location

### Save Configuration

1. Control Panel → Adjust parameters
2. Click **💾 Save Config**
3. Configuration saved for next session

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Q` | Quit application |
| `Ctrl+S` | Save configuration |
| `Ctrl+E` | Export metrics |
| `Ctrl+L` | Clear terminal |
| `Tab` | Switch between tabs |

## Troubleshooting

### "Qallow binary not found"

```bash
# Check if binary exists
ls -la /root/Qallow/build/qallow*

# Make executable
chmod +x /root/Qallow/build/qallow*
```

### "Failed to start VM"

```bash
# Test binary directly
/root/Qallow/build/qallow phase 14 --ticks=100

# Check permissions
ls -la /root/Qallow/build/qallow
```

### "Terminal output not showing"

- Ensure Qallow binary is executable
- Check system logs for errors
- Try running binary directly first

### "Metrics not updating"

- Start web dashboard: `cd /root/Qallow/ui && python3 dashboard.py`
- Check API URL in settings
- Verify network connectivity

### "Application crashes"

```bash
# Run with debug output
RUST_LOG=debug ./target/release/qallow-native

# Check system resources
free -h
df -h
```

## Tips & Tricks

### 1. Monitor Multiple Phases

- Run Phase 13 first for baseline
- Then run Phase 14 for coherence
- Finally run Phase 15 for convergence

### 2. Optimize Performance

- Use CUDA build for faster execution
- Adjust tick count based on your system
- Monitor GPU usage in Metrics tab

### 3. Save Configurations

- Save different configurations for different phases
- Export metrics after each run
- Review audit logs for insights

### 4. Batch Operations

- Create shell script to run multiple phases
- Use exported metrics for analysis
- Automate with cron jobs

## Performance Tips

### For CPU-Only Systems

- Use Phase 13 with 400 ticks
- Reduce tick count if system is slow
- Monitor CPU usage in Metrics

### For GPU Systems

- Use CUDA build for 2-3x speedup
- Phase 14 with 600 ticks recommended
- Monitor GPU memory in Metrics

### For Low-Memory Systems

- Use Phase 13 (lowest memory)
- Reduce tick count
- Close other applications

## Next Steps

1. **Read Full Documentation**: See README.md
2. **Explore All Tabs**: Familiarize with interface
3. **Run Different Phases**: Try all three phases
4. **Export Data**: Save metrics for analysis
5. **Customize Settings**: Adjust parameters

## Getting Help

### Documentation
- README.md - Full documentation
- This file - Quick start guide
- /root/Qallow/docs/ - Qallow documentation

### Logs
- Terminal tab - Live output
- Audit Log tab - Event history
- ~/.qallow/logs/ - Application logs

### Support
- Check troubleshooting section
- Review Qallow documentation
- Check system logs: `journalctl -xe`

## Quick Reference

```bash
# Build
cd /root/Qallow/native_app && cargo build --release

# Run
./target/release/qallow-native

# Test binary
/root/Qallow/build/qallow phase 14 --ticks=100

# Check logs
tail -f ~/.qallow/logs/app.log

# Clean build
cargo clean && cargo build --release
```

---

**Ready to go!** 🚀

Start the app and explore the Qallow Quantum-Photonic AGI System!

```bash
./target/release/qallow-native
```

