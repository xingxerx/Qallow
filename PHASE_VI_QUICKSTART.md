# Phase VI – Quick Start Guide

## One-Command Operation

All Phase 6 functionality is integrated into the unified `qallow` command.

---

## Commands

### 1. Verify System Health

```bash
qallow verify
```

**Output**:
```
Status: PASS
Coherence:     0.999200 (min: 0.995000) [OK]
Decoherence:   0.000010 (max: 0.001000) [OK]
Ethics Score:  2.998400 (min: 2.990000) [OK]
Sandbox:       ACTIVE
Telemetry:     ACTIVE
Ethics:        ENFORCED
```

### 2. Start Live Interface

```bash
qallow live
```

**Output**:
```
[LIVE] Starting Phase 6 - Live Interface and External Data Integration
[LIVE] Ingestion manager initialized with 4 streams
[LIVE] Streams configured and ready for data ingestion
[LIVE] - telemetry_primary: http://localhost:9000/telemetry
[LIVE] - sensor_coherence: http://localhost:9001/coherence
[LIVE] - sensor_decoherence: http://localhost:9002/decoherence
[LIVE] - feedback_hitl: http://localhost:9003/feedback
[LIVE] Running VM with live data integration...
```

### 3. Run Standard VM

```bash
qallow run
```

### 4. Run Benchmark

```bash
qallow bench
```

### 5. Run Governance Audit

```bash
qallow govern
```

### 6. Show Help

```bash
qallow help
```

---

## Workflow

### Step 1: Verify System

```bash
qallow verify
```

Wait for `Status: PASS` before proceeding.

### Step 2: Start Live Interface

```bash
qallow live
```

This initializes:
- Ingestion manager
- 4 default data streams
- Network adapters
- VM execution with live data

### Step 3: Monitor Telemetry

In another terminal:

```bash
tail -f qallow_stream.csv
```

### Step 4: Check Logs

```bash
cat qallow_bench.log
```

---

## Configuration

### Add Custom Stream

Edit `io/sensors.json`:

```json
{
  "name": "my_sensor",
  "type": "telemetry",
  "endpoint": "http://localhost:9005/data",
  "adapter": "net_adapter",
  "enabled": true,
  "poll_interval_ms": 100
}
```

### Modify Thresholds

Edit `io/sensors.json`:

```json
"thresholds": {
  "coherence_min": 0.995,
  "decoherence_max": 0.001,
  "ethics_min": 2.99,
  "stability_duration_seconds": 300
}
```

---

## Data Streams

### Default Streams

1. **telemetry_primary** - Primary telemetry (port 9000)
2. **sensor_coherence** - Coherence measurement (port 9001)
3. **sensor_decoherence** - Decoherence measurement (port 9002)
4. **feedback_hitl** - Human feedback (port 9003)

### Packet Format

```json
{
  "timestamp": 1697653200,
  "type": "telemetry",
  "value": 0.832,
  "confidence": 0.95,
  "source": "http_endpoint",
  "metadata": "coherence_measurement"
}
```

---

## Troubleshooting

### Verify Fails

**Issue**: `Status: FAIL`

**Solution**: 
- Check coherence: should be ≥ 0.995
- Check decoherence: should be < 0.001
- Check ethics: should be ≥ 2.99

### Live Interface Crashes

**Issue**: Command exits unexpectedly

**Solution**:
- Ensure `io/sensors.json` exists
- Check that endpoints are accessible
- Verify network connectivity

### No Data Ingestion

**Issue**: Streams show 0 packets

**Solution**:
- Verify endpoints are running
- Check network connectivity
- Review adapter logs

---

## Performance

### Typical Metrics

- **Coherence**: 0.9992
- **Decoherence**: 0.000010
- **Ethics Score**: 2.9984
- **Runtime**: ~1ms per cycle
- **Throughput**: 256 nodes per overlay

### Optimization

For better performance:
- Reduce `poll_interval_ms` in sensors.json
- Increase `INGEST_BUFFER_SIZE` in ingest.h
- Enable CUDA: `qallow build` (with CUDA support)

---

## Files

### Configuration

- `io/sensors.json` - Stream configuration

### Logs

- `qallow_stream.csv` - Telemetry data
- `qallow_bench.log` - Benchmark results
- `logs/audit_*.json` - Audit logs (every 60s)

### Snapshots

- `snapshots/phase6_safe.bin` - State snapshots
- `snapshots/phase6.sig` - Integrity hashes

---

## Next Phase

When conditions are met for ≥ 5 minutes:
- Coherence > 0.995
- Decoherence < 0.001
- Ethics ≥ 2.99
- No violations

→ **Phase 7: Distributed Swarm Nodes**

---

**Status**: ✅ Phase 6 Ready

