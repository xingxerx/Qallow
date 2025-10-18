# Qallow Phase 13: Closed-Loop Ethics System

## Overview

This system bridges Qallow's ethics algorithm with **hardware-verified telemetry**, creating a closed-loop that responds to real, measurable data instead of static inputs.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    HARDWARE SOURCES                         │
│  • CPU temperature    • System load     • Build logs        │
│  • Memory pressure    • ECC errors      • Human feedback    │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│           collect_signals.py (Python Collector)             │
│  Polls hardware, normalizes to [0,1], outputs timestamped   │
│  → /data/telemetry/current_signals.txt                      │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│           ethics_feed.c (C Ingestion Layer)                 │
│  Validates, clamps, logs incoming signals                   │
│  → ethics_metrics_t {safety, clarity, human}                │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│         ethics_core.c (Decision Engine)                     │
│  Computes weighted score, applies thresholds                │
│  → PASS/FAIL decision + adaptive learning                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. Hardware Collector (`python/collect_signals.py`)

**Purpose:** Translate raw system metrics into normalized ethics signals.

**Metrics:**

| Dimension | Sources | Normalization |
|-----------|---------|---------------|
| **Safety** | CPU temp, load avg, memory | 1.0 = healthy, 0.0 = critical |
| **Clarity** | Build errors, warnings, tests | 1.0 = clean, 0.0 = broken |
| **Human** | Operator feedback file | Direct [0,1] input |

**Output format:**
```
# 1729267890
0.950 0.920 0.880 0.990 0.950 0.970 0.980 0.750 0.800 0.780
```
- Line 1: Timestamp (Unix epoch)
- Line 2: 10 space-separated floats (3 safety + 4 clarity + 3 human)

**Usage:**
```bash
# Single collection
python3 python/collect_signals.py

# Continuous daemon (5s interval)
python3 python/collect_signals.py --loop &
```

---

### 2. Ingestion Layer (`algorithms/ethics_feed.c`)

**Function:** `int ethics_ingest_signal(const char *path, ethics_metrics_t *metrics)`

**Responsibilities:**
- Parse signal file
- Validate range [0,1]
- Clamp out-of-range values
- Log ingestion with timestamp

**Integration:**
```c
#include "ethics_core.h"

ethics_metrics_t metrics;
if (ethics_ingest_signal("/root/Qallow/data/telemetry/current_signals.txt", &metrics)) {
    // Use metrics for scoring
}
```

---

### 3. Test Programs

#### `ethics_test` (Original)
Static inputs for baseline testing:
```bash
cd algorithms
./ethics_test
```

#### `ethics_test_feed` (Closed-Loop)
Reads hardware signals and demonstrates real-time ethics evaluation:
```bash
cd algorithms
./ethics_test_feed
```

**Sample output:**
```
========================================
Qallow Ethics Test - Closed-Loop Mode
========================================

[1] Model load: config
weights  -> safety: 1.100 clarity: 1.000 human: 0.900
thresholds -> safety: 0.700 clarity: 0.650 human: 0.600 total: 2.100

[2] Ingesting hardware signals...
[ethics_feed] Ingested at Fri Oct 18 14:23:12 2025
  Safety:  0.950
  Clarity: 0.920
  Human:   0.750
[2] ✓ Hardware signals loaded

[3] Computing ethics score...
  Weighted components:
    Safety:  0.950 × 1.10 = 1.045
    Clarity: 0.920 × 1.00 = 0.920
    Human:   0.750 × 0.90 = 0.675
  Total score: 2.640
  Threshold:   2.100
  Result:      ✓ PASS

[4] Applying adaptive feedback...
  Model after adaptation:
  weights  -> safety: 1.110 clarity: 1.010 human: 0.910
  Posterior trust: 0.692

========================================
Test complete: SYSTEM ETHICAL
========================================
```

---

## Quick Start

### 1. Setup
```bash
# Create data directories
mkdir -p /root/Qallow/data/telemetry

# Set initial human feedback (optional)
echo "0.75" > /root/Qallow/data/human_feedback.txt

# Make scripts executable
chmod +x /root/Qallow/scripts/test_closed_loop.sh
```

### 2. Build
```bash
cd /root/Qallow/algorithms
make clean && make
```

### 3. Run Closed-Loop Test
```bash
/root/Qallow/scripts/test_closed_loop.sh
```

This will:
1. Collect hardware signals
2. Build ethics system
3. Run test with real data
4. Display results

---

## Integration with Qallow Unified

### Option A: Periodic Collection (Recommended)

Add to main loop or systemd timer:
```bash
# Every 5 seconds, update signals
while true; do
    python3 /root/Qallow/python/collect_signals.py
    sleep 5
done
```

### Option B: On-Demand Collection

Call before ethics-critical operations:
```c
// In your main.c or decision point
system("python3 python/collect_signals.py");

ethics_metrics_t metrics;
if (ethics_ingest_signal("/root/Qallow/data/telemetry/current_signals.txt", &metrics)) {
    // Proceed with ethics scoring
}
```

### Option C: Direct Integration

Add to CMakeLists.txt or build system:
```cmake
set(ETHICS_SOURCES
    algorithms/ethics_core.c
    algorithms/ethics_learn.c
    algorithms/ethics_bayes.c
    algorithms/ethics_feed.c
)
```

Then in your code:
```c
#include "ethics_core.h"

int ethics_ingest_signal(const char *path, ethics_metrics_t *metrics);

// In runtime loop
ethics_metrics_t metrics;
ethics_ingest_signal("/root/Qallow/data/telemetry/current_signals.txt", &metrics);

ethics_model_t model;
ethics_model_load(&model, "config/weights.json", "config/thresholds.json");

ethics_score_details_t details;
double score = ethics_score_core(&model, &metrics, &details);
int pass = ethics_score_pass(&model, &metrics, &details);

if (!pass) {
    // Handle ethics violation
    qallow_halt_operation("Ethics threshold not met");
}
```

---

## Validation & Verification

### Test Signal Freshness
```bash
stat -c %Y /root/Qallow/data/telemetry/current_signals.txt
date +%s
# Difference should be < 10s for real-time operation
```

### Monitor Collection Logs
```bash
tail -f /root/Qallow/data/telemetry/collection.log
```

### Verify Signal Range
```bash
jq . /root/Qallow/data/telemetry/current_signals.json
# All values should be [0,1]
```

### Stress Test
```bash
# Simulate thermal event
# (CPU temp rises → safety drops)
stress-ng --cpu 8 --timeout 30s &
python3 python/collect_signals.py
cat /root/Qallow/data/telemetry/current_signals.txt
```

---

## Security Considerations

1. **Signal Validation**
   - All inputs clamped to [0,1]
   - Timestamp verification available
   - Out-of-range logged but not fatal

2. **File Permissions**
   ```bash
   chmod 600 /root/Qallow/data/telemetry/current_signals.txt
   chmod 600 /root/Qallow/data/human_feedback.txt
   ```

3. **Audit Trail**
   - Use `ethics_log_decision()` to record all ethics evaluations
   - Logs include timestamp, score, action taken

4. **Freshness Checks**
   - `ethics_verify_freshness()` ensures data < 5s old
   - Prevents stale data from being used in critical decisions

---

## Troubleshooting

### Signals not updating
```bash
# Check collector
python3 python/collect_signals.py
echo $?  # Should be 0

# Check permissions
ls -l /root/Qallow/data/telemetry/
```

### Build errors
```bash
cd /root/Qallow/algorithms
make clean
make CFLAGS="-std=c11 -O2 -Wall -Wextra -I../core/include"
```

### Missing thermal sensors
Edit `collect_signals.py`:
```python
# Use fallback if no hardware sensors
thermal_score = 0.95  # Assume OK
```

---

## Future Enhancements

1. **GPU Metrics** (if CUDA enabled)
   ```python
   def gpu_safety():
       temp = nvidia_smi_query("temperature.gpu")
       return max(0.0, 1.0 - (temp - 50) / 40.0)
   ```

2. **Network Telemetry**
   - Expose `/metrics` endpoint
   - Prometheus integration
   - Grafana dashboard

3. **Multi-Source Fusion**
   - Bayesian aggregation of multiple sensors
   - Confidence intervals
   - Outlier detection

4. **Cryptographic Signatures**
   - Sign signal files with hardware TPM
   - Verify integrity before ingestion

---

## Files Created

```
/root/Qallow/
├── algorithms/
│   ├── ethics_feed.c           # Ingestion layer
│   ├── ethics_test_feed.c      # Closed-loop test
│   └── Makefile                # Updated build
├── python/
│   └── collect_signals.py      # Hardware collector
├── scripts/
│   └── test_closed_loop.sh     # Integration test
└── docs/
    └── PHASE13_CLOSED_LOOP.md  # This document
```

---

## References

- **Ethics Core:** `/root/Qallow/algorithms/ethics_core.c`
- **Original Test:** `/root/Qallow/algorithms/ethics_test.c`
- **Config:** `/root/Qallow/config/{weights,thresholds}.json`
- **Telemetry Output:** `/root/Qallow/data/telemetry/`

---

**Status:** ✓ Ready for deployment  
**Last Updated:** October 18, 2025  
**Phase:** 13 - Autonomous Ethics with Hardware Verification
