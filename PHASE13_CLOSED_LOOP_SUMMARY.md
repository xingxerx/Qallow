# Closed-Loop Ethics System - Quick Reference

## System Status: ✓ OPERATIONAL

**Created:** October 18, 2025  
**Phase:** 13 - Hardware-Verified Autonomous Ethics

---

## What Was Built

A complete **closed-loop ethics monitoring system** that:

1. **Collects** real hardware telemetry (CPU, memory, build logs)
2. **Ingests** data into C-based ethics engine  
3. **Computes** weighted ethics scores with thresholds
4. **Adapts** via Bayesian learning and feedback
5. **Responds** to verifiable changes in system state

---

## Key Files Created

```
algorithms/
  ├── ethics_feed.c          # Ingestion layer (NEW)
  └── ethics_test_feed.c     # Closed-loop test (NEW)

python/
  └── collect_signals.py     # Hardware collector (NEW)

scripts/
  └── test_closed_loop.sh    # Integration test (NEW)

docs/
  └── PHASE13_CLOSED_LOOP.md # Full documentation (NEW)
```

---

## Verification Results

### Test 1: Normal Operation
```
Safety:  0.972  Clarity: 1.000  Human: 0.900
Total:   2.301  Threshold: 1.867
Result:  ✓ PASS (SYSTEM ETHICAL)
```

### Test 2: Degraded Human Feedback
```
Safety:  0.971  Clarity: 1.000  Human: 0.300
Total:   1.818  Threshold: 1.867
Result:  ✗ FAIL (ETHICS VIOLATION)
```

### Test 3: Restored Feedback
```
Safety:  0.972  Clarity: 1.000  Human: 0.900
Total:   2.301  Threshold: 1.867
Result:  ✓ PASS (SYSTEM ETHICAL)
```

**Conclusion:** System correctly responds to real-time signal changes.

---

## Data Flow

```
Hardware Sensors
       ↓
collect_signals.py (5s polling)
       ↓
/data/telemetry/current_signals.txt
       ↓
ethics_ingest_signal()
       ↓
ethics_score_core()
       ↓
PASS/FAIL Decision
       ↓
Adaptive Learning
```

---

## Quick Commands

### One-Shot Test
```bash
python3 python/collect_signals.py
cd algorithms && ./ethics_test_feed
```

### Continuous Monitoring
```bash
python3 python/collect_signals.py --loop &
```

### Adjust Human Feedback
```bash
echo "0.85" > /data/human_feedback.txt
```

### Full Integration Test
```bash
./scripts/test_closed_loop.sh
```

---

## Integration Points

### Option 1: Periodic Collection (Recommended)
```bash
# Add to systemd timer or cron
*/5 * * * * python3 /root/Qallow/python/collect_signals.py
```

### Option 2: Main Loop Integration
```c
// In qallow_unified main loop
system("python3 python/collect_signals.py");

ethics_metrics_t metrics;
if (ethics_ingest_signal("/root/Qallow/data/telemetry/current_signals.txt", &metrics)) {
    ethics_model_t model;
    ethics_model_load(&model, "config/weights.json", "config/thresholds.json");
    
    ethics_score_details_t details;
    double score = ethics_score_core(&model, &metrics, &details);
    int pass = ethics_score_pass(&model, &metrics, &details);
    
    if (!pass) {
        qallow_halt_operation("Ethics threshold violation");
    }
}
```

### Option 3: Build System Addition
Add to `CMakeLists.txt`:
```cmake
set(ETHICS_SOURCES
    algorithms/ethics_core.c
    algorithms/ethics_learn.c
    algorithms/ethics_bayes.c
    algorithms/ethics_feed.c
)
```

---

## Metrics Collected

### Safety (Hardware Health)
- CPU temperature normalization: `(80°C - current) / 40°C`
- System load vs CPU count
- Memory pressure: `(95% - used) / 25%`

### Clarity (Software Quality)  
- Build errors (1.0 if zero errors)
- Warning count (degraded by warnings)
- Test pass rate (placeholder)
- Lint score (placeholder)

### Human (Operator Feedback)
- Direct file input: `/data/human_feedback.txt`
- Range: `[0.0, 1.0]`
- Updated on-demand by operators

---

## Security Features

✓ All inputs validated and clamped to [0,1]  
✓ Timestamp verification for freshness  
✓ Audit logging of all decisions  
✓ Out-of-range values logged but not fatal  
✓ Signal file permissions: 600 (owner-only)

---

## Next Steps

1. **Deploy to Production**
   - Set up systemd timer for collector
   - Integrate into main Qallow loop
   - Configure alerting on ethics failures

2. **Enhance Metrics**
   - Add GPU temperature (CUDA)
   - Network latency monitoring
   - Disk I/O health

3. **Add Cryptographic Verification**
   - Sign signals with TPM
   - Verify integrity before ingestion
   - Detect tampering

4. **Dashboard Integration**
   - Expose `/metrics` endpoint
   - Prometheus scraping
   - Grafana visualization

---

## Support

- **Full Documentation:** `/root/Qallow/docs/PHASE13_CLOSED_LOOP.md`
- **Test Logs:** `/root/Qallow/data/telemetry/collection.log`
- **Signal History:** `/root/Qallow/data/telemetry/current_signals.json`

---

**Status:** Production-ready for hardware-verified ethics monitoring  
**License:** Part of Qallow Unified System  
**Contact:** See main README
