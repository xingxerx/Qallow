# Qallow Telemetry & Analytics Quick Reference

## 🎯 Overview

Qallow generates rich telemetry data during execution that can be analyzed for:
- **Ethics & Safety Auditing** - Track S, C, H metrics and reality drift
- **Quantum Research** - Analyze coherence, decoherence, and overlay stability
- **Performance Benchmarking** - Compare CPU vs CUDA, measure convergence
- **AGI Development** - Study emergent behaviors and cognitive patterns

## 📊 Available Tools

### 1. **Post-Run Analysis** - `analyze_telemetry_simple.py`
Comprehensive analysis of completed runs with detailed statistics and JSON reports.

```bash
# Run analysis after Qallow execution
python3 scripts/analyze_telemetry_simple.py

# Or use the full-featured version (requires matplotlib)
python3 scripts/analyze_telemetry.py
```

**Output:**
- Console statistics and metrics
- JSON report: `data/logs/analysis_report_TIMESTAMP.json`
- Visualizations (full version): `data/logs/telemetry_analysis_TIMESTAMP.png`

### 2. **Live Monitoring** - `monitor_live.py`
Real-time dashboard that updates every 2 seconds during Qallow execution.

```bash
# Terminal 1: Start Qallow
./build/qallow

# Terminal 2: Start live monitor
python3 scripts/monitor_live.py
```

**Features:**
- Real-time ethics scores (S, C, H, E)
- Live quantum coherence tracking
- ASCII bar graphs with color coding
- Instant violation alerts
- 2-second refresh rate

## 📁 Data Files

All telemetry is stored in `data/logs/`:

| File | Content | Use Case |
|------|---------|----------|
| `telemetry_stream.csv` | Quantum coherence & overlay stability per tick | Quantum analysis |
| `phase13.csv` | Ethics metrics (S, C, H, E) and drift | Ethics auditing |
| `phase12.csv` | Additional phase data | Extended analysis |
| `qallow_bench.log` | Performance benchmarks | Optimization |
| `analysis_report_*.json` | Generated analysis reports | Historical tracking |

## 📈 Key Metrics Explained

### Ethics Metrics (Phase 13)
- **Safety (S)**: System stability (threshold: > 0.80)
- **Clarity (C)**: Decision transparency (target: > 0.85)
- **Human (H)**: Human value alignment (target: > 0.90)
- **Total Ethics (E)**: S + C + H - Δ (excellent: > 2.7)
- **Reality Drift (Δ)**: System divergence (limit: < 0.25)

### Quantum Metrics (Telemetry Stream)
- **Orbital/River/Mycelial**: Individual overlay stability
- **Global Stability**: Combined system coherence (target: > 0.95)
- **Coherence**: Average quantum state fidelity
- **Decoherence**: Quantum state degradation (lower is better)

## 🔍 Analysis Workflow

### Standard Analysis Cycle
```bash
# 1. Run Qallow to generate data
./build/qallow

# 2. Analyze the telemetry
python3 scripts/analyze_telemetry_simple.py

# 3. Review the JSON report
cat data/logs/analysis_report_*.json | jq '.'

# 4. Compare with previous runs
diff data/logs/analysis_report_*.json
```

### Real-Time Monitoring
```bash
# In separate terminals:
./build/qallow              # Terminal 1
python3 scripts/monitor_live.py  # Terminal 2
```

## 🎨 Visualization (Full Version)

If you have matplotlib installed:
```bash
# Install dependencies
pip install matplotlib pandas seaborn

# Run with visualizations
python3 scripts/analyze_telemetry.py
```

Generates multi-panel dashboard showing:
- Ethics trends over time
- Reality drift tracking
- Overlay stability evolution
- Coherence/decoherence graphs
- Statistical summary panel

## 🔬 Research Applications

### 1. Ethics Research
```bash
# Extract ethics violations
grep "VIOLATION" data/logs/analysis_report_*.json

# Compare ethics scores across runs
jq '.ethics.average_ethics_score' data/logs/analysis_report_*.json
```

### 2. Quantum Performance
```bash
# Check coherence trends
cut -d',' -f2,3,4 data/logs/telemetry_stream.csv | head -20

# Compare CUDA vs CPU performance
grep "mode" data/logs/telemetry_stream.csv | sort | uniq -c
```

### 3. Continuous Integration
```bash
# Automated testing workflow
./build/qallow && \
python3 scripts/analyze_telemetry_simple.py && \
jq '.status' data/logs/analysis_report_*.json | grep -q "healthy" && \
echo "✅ CI PASSED" || echo "❌ CI FAILED"
```

## 🚀 Advanced Usage

### Custom Analysis Scripts
```python
from scripts.analyze_telemetry_simple import QallowAnalyzer

analyzer = QallowAnalyzer()
ethics_stats = analyzer.analyze_ethics()
quantum_stats = analyzer.analyze_quantum()

# Your custom analysis here
print(f"Ethics score: {ethics_stats['Total Ethics (E)']['mean']}")
```

### Export to Other Tools
```bash
# Export to CSV for Excel/Sheets
cp data/logs/phase13.csv ~/analysis/ethics_data.csv

# Convert JSON report to YAML
python3 -c "import json, yaml; print(yaml.dump(json.load(open('data/logs/analysis_report_*.json'))))"

# Send metrics to monitoring system
curl -X POST https://monitoring.example.com/metrics \
  -d @data/logs/analysis_report_latest.json
```

## 📊 Interpreting Results

### Healthy System Indicators
✅ Ethics score > 2.7  
✅ Safety (S) > 0.80  
✅ Reality drift < 0.05  
✅ Global stability > 0.95  
✅ Zero violations  

### Warning Signs
⚠️ Ethics score < 2.5  
⚠️ Safety violations present  
⚠️ Drift > 0.15  
⚠️ Coherence < 0.90  

### Critical Issues
🔴 Safety score < 0.70  
🔴 Reality drift > 0.25  
🔴 Multiple violations  
🔴 Decoherence > 0.20  

## 🛠️ Troubleshooting

### No Data Available
```bash
# Check if Qallow generated logs
ls -lh data/logs/*.csv

# Run Qallow first if empty
./build/qallow
```

### Analysis Script Errors
```bash
# Check Python version (need 3.8+)
python3 --version

# Verify CSV files are not corrupted
head -5 data/logs/telemetry_stream.csv
```

### Matplotlib Issues
```bash
# Use lightweight version instead
python3 scripts/analyze_telemetry_simple.py

# Or fix matplotlib
pip install --force-reinstall matplotlib
```

## 📚 Next Steps

1. **Run your first analysis**: `python3 scripts/analyze_telemetry_simple.py`
2. **Try live monitoring**: `python3 scripts/monitor_live.py`
3. **Review the JSON report**: Look for recommendations
4. **Experiment with parameters**: Adjust node count, ticks, etc.
5. **Track trends**: Run multiple times and compare results

## 🤝 Integration Points

- **DeepSeek Baseline**: Feed ethics data for cognitive auditing
- **Quantum Orchestrator**: Use coherence metrics to optimize circuits
- **Memory System**: Store and learn from historical metrics
- **REST API**: Expose real-time metrics via API endpoints
- **CI/CD Pipelines**: Automated quality gates based on metrics

---

**For more info:**
- Main README: `README.md`
- Playbook: `.github/copilot-instructions.md`
- Build Guide: `BUILD_RUN_GUIDE.md`
