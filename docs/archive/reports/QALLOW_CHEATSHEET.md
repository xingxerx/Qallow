# Qallow Unified Command - Cheatsheet

## One Command to Rule Them All

```bash
./qallow <command> [target]
```

---

## Commands at a Glance

| Command | Usage | Purpose |
|---------|-------|---------|
| `build` | `./qallow build [cpu\|cuda\|all]` | Compile the system |
| `run` | `./qallow run [cpu\|cuda]` | Execute simulation |
| `bench` | `./qallow bench [cpu\|cuda\|all]` | Run benchmarks |
| `telemetry` | `./qallow telemetry [stream\|bench\|adapt]` | View data |
| `status` | `./qallow status` | System overview |
| `clean` | `./qallow clean` | Remove builds |
| `help` | `./qallow help [command]` | Show help |

---

## Most Common Usage

```bash
# Build everything
./qallow build all

# Run CPU version
./qallow run cpu

# Benchmark both
./qallow bench all

# Check status
./qallow status

# View telemetry
./qallow telemetry stream
```

---

## Build Commands

```bash
./qallow build              # CPU only (default)
./qallow build cpu          # CPU only
./qallow build cuda         # CUDA only
./qallow build all          # Both CPU and CUDA
```

---

## Run Commands

```bash
./qallow run                # CPU only (default)
./qallow run cpu            # CPU only
./qallow run cuda           # CUDA only
```

---

## Benchmark Commands

```bash
./qallow bench              # CPU only (default)
./qallow bench cpu          # CPU only
./qallow bench cuda         # CUDA only
./qallow bench all          # Both CPU and CUDA
```

---

## Telemetry Commands

```bash
./qallow telemetry          # Stream data (default)
./qallow telemetry stream   # Real-time tick data
./qallow telemetry bench    # Benchmark history
./qallow telemetry adapt    # Adaptive state
./qallow telemetry all      # All telemetry
```

---

## Output Files

| File | Contains |
|------|----------|
| `qallow_stream.csv` | Real-time tick data |
| `qallow_bench.log` | Benchmark history |
| `adapt_state.json` | Adaptive parameters |
| `build/qallow.exe` | CPU executable |
| `build/qallow_cuda.exe` | CUDA executable |

---

## Quick Workflows

### Development
```bash
./qallow build cpu
./qallow run cpu
./qallow status
```

### Benchmarking
```bash
./qallow build all
./qallow bench all
./qallow telemetry bench
```

### Full Test
```bash
./qallow clean
./qallow build all
./qallow run cpu
./qallow bench all
./qallow status
```

---

## Status Output

```
Build Status:
[OK] CPU build ready (219.5 KB)
[OK] CUDA build ready (221.5 KB)

Telemetry Files:
[OK] Stream data: 2 lines
[OK] Benchmark log: 16 lines

System Info:
[CPU] AMD Ryzen 7 7800X3D 8-Core Processor
[CORES] 8
[RAM] 31.2 GB
[GPU] NVIDIA GeForce RTX 5080
```

---

## Telemetry Output

### Stream Data (qallow_stream.csv)
```csv
tick,orbital,river,mycelial,global,deco,mode
0,0.9984,0.9982,0.9984,0.9992,0.00001,CPU
```

### Benchmark Log (qallow_bench.log)
```
timestamp,compile_ms,run_ms,deco,global,mode
2025-10-18 07:56:49,0.0,1.00,0.00001,0.9992,CPU
```

### Adaptive State (adapt_state.json)
```json
{
  "target_ms": 50.0,
  "threads": 4,
  "learning_rate": 0.0034,
  "human_score": 0.8
}
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Command not found | Use `./qallow_unified.ps1` or `./qallow.bat` |
| Build fails | Run `./qallow clean` then `./qallow build cpu` |
| CUDA fails | Check `nvidia-smi` and CUDA Toolkit installation |
| No telemetry | Run `./qallow run cpu` first |

---

## Performance Comparison

```bash
# Run both versions
./qallow run cpu
./qallow run cuda

# Compare benchmarks
./qallow bench all

# View results
./qallow telemetry bench
```

---

## System Requirements

- Windows 10/11
- Visual Studio 2022 Build Tools
- CUDA Toolkit 13.0 (for CUDA builds)
- NVIDIA GPU (for CUDA execution)

---

## File Locations

```
Qallow/
├── qallow_unified.ps1       ← Main command
├── qallow.bat               ← Batch wrapper
├── qallow.cmd               ← CMD wrapper
├── build/
│   ├── qallow.exe           ← CPU executable
│   └── qallow_cuda.exe      ← CUDA executable
├── qallow_stream.csv        ← Real-time data
├── qallow_bench.log         ← Benchmark history
└── adapt_state.json         ← Adaptive state
```

---

## Tips & Tricks

1. **Alias it:** Add to PowerShell profile
   ```powershell
   Set-Alias qallow ./qallow_unified.ps1
   ```

2. **Add to PATH:** Run from anywhere
   ```bash
   $env:PATH += ";C:\path\to\Qallow"
   ```

3. **Batch operations:**
   ```bash
   ./qallow build all && ./qallow bench all && ./qallow status
   ```

4. **Monitor in real-time:**
   ```bash
   ./qallow run cpu
   # In another terminal:
   ./qallow telemetry stream
   ```

---

## Status Codes

| Code | Meaning |
|------|---------|
| `[OK]` | Success |
| `[PENDING]` | Not yet generated |
| `[MISSING]` | File not found |
| `[ERROR]` | Operation failed |

---

## Next Steps

```bash
# 1. Build
./qallow build all

# 2. Run
./qallow run cpu

# 3. Benchmark
./qallow bench all

# 4. Monitor
./qallow status

# 5. Analyze
./qallow telemetry all
```

---

**Version:** 1.0.0  
**Status:** 🟢 Production Ready

