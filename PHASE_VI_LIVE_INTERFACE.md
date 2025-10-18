# Phase VI – Live Interface and External Data Integration

**Goal**: Turn the working Qallow VM + CUDA core into an online adaptive system that ingests and reacts to real-world data while keeping its ethics and sandbox intact.

---

## 1. System Checkpoint

Before expanding to Phase 6, verify system integrity:

```bash
qallow verify
```

**Expected Output**:
- Coherence ≈ 0.999
- Decoherence < 0.001
- Ethics E ≥ 2.99
- Status: PASS

---

## 2. Data Ingress Layer

### Structure

```
io/
 ├─ sensors.json         # Endpoint configurations
 └─ adapters/
      ├─ net_adapter.c   # HTTP/REST endpoints
      ├─ serial_adapter.c # Serial port data
      └─ sim_adapter.c   # Internal simulation
```

### Normalized Data Packet Format

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

### Packet Types

- `INGEST_TYPE_TELEMETRY` - System telemetry
- `INGEST_TYPE_SENSOR` - Sensor measurements
- `INGEST_TYPE_CONTROL` - Operator commands
- `INGEST_TYPE_FEEDBACK` - Human-in-the-loop scores
- `INGEST_TYPE_OVERRIDE` - Emergency overrides

---

## 3. Pipeline Flow

```
[Sensor/Feed] → adapters → ingest.c → pocket.cu → ethics_core.c → telemetry.csv
```

**Key Components**:

1. **Adapters** - Convert external signals to normalized JSON
2. **Ingest Manager** - Circular buffer for packet management
3. **Pocket Dimension (CUDA)** - Short-cycle simulations with boundary conditions
4. **Ethics Core** - Validates E = S + C + H before execution
5. **Telemetry** - Logs all results to CSV

---

## 4. Live Interface Commands

### Verify System Health

```bash
qallow verify
```

Checks:
- Coherence measurement
- Decoherence level
- Ethics score
- Sandbox status
- Telemetry status

### Start Live Interface

```bash
qallow live
```

Initializes:
- Ingestion manager with 4 default streams
- Network adapters for HTTP endpoints
- Simulation adapter for testing
- VM execution with live data integration

### Stream Configuration

Edit `io/sensors.json` to add/modify streams:

```json
{
  "name": "telemetry_primary",
  "type": "telemetry",
  "endpoint": "http://localhost:9000/telemetry",
  "adapter": "net_adapter",
  "enabled": true,
  "poll_interval_ms": 100
}
```

---

## 5. Adaptive Learning Loop

In `backend/cpu/adaptive.c`:

```c
Δθ = η * (HITL_score – predicted_score);
```

Where:
- η ≈ 0.003 (learning rate)
- HITL_score = human evaluation
- predicted_score = model prediction

This ties human feedback directly to model updates without breaking sandbox isolation.

---

## 6. Audit & Snapshot

Every 60 seconds:

1. Write `/logs/audit_YYYYMMDD.json`
2. Freeze state → `/snapshots/phase6_safe.bin`
3. Hash integrity: `sha256sum phase6_safe.bin > phase6.sig`

---

## 7. Thresholds for Phase 7 Promotion

When the following conditions are met for ≥ 5 minutes:

- External → Internal coherence > 0.995
- Decoherence < 0.001
- Ethics score ≥ 2.99
- No violations detected

**Then**: Promote to **Phase 7: Distributed Swarm Nodes**

---

## 8. Quick Start

### 1. Verify System

```bash
qallow verify
```

### 2. Start Live Interface

```bash
qallow live
```

### 3. Monitor Telemetry

```bash
tail -f qallow_stream.csv
```

### 4. Check Logs

```bash
cat qallow_bench.log
```

---

## 9. Architecture

### Ingestion Manager

```c
typedef struct {
    ingest_stream_t streams[INGEST_MAX_STREAMS];
    ingest_packet_t packet_buffer[INGEST_BUFFER_SIZE];
    int buffer_head, buffer_tail, buffer_count;
    uint64_t total_packets, total_dropped;
    int paused, running;
} ingest_manager_t;
```

### Stream Configuration

```c
typedef struct {
    char name[64];
    char endpoint[256];
    int enabled;
    uint64_t packets_received;
    uint64_t packets_dropped;
    double last_value;
    time_t last_update;
} ingest_stream_t;
```

---

## 10. API Reference

### Ingestion Manager

```c
void ingest_init(ingest_manager_t* mgr);
void ingest_cleanup(ingest_manager_t* mgr);

int ingest_add_stream(ingest_manager_t* mgr, const char* name, const char* endpoint);
int ingest_remove_stream(ingest_manager_t* mgr, const char* name);
int ingest_enable_stream(ingest_manager_t* mgr, const char* name);
int ingest_disable_stream(ingest_manager_t* mgr, const char* name);
int ingest_pause_all(ingest_manager_t* mgr);
int ingest_resume_all(ingest_manager_t* mgr);

int ingest_push_packet(ingest_manager_t* mgr, const ingest_packet_t* packet);
int ingest_pop_packet(ingest_manager_t* mgr, ingest_packet_t* packet);
int ingest_peek_packet(ingest_manager_t* mgr, ingest_packet_t* packet);
int ingest_packet_count(ingest_manager_t* mgr);

void ingest_print_stats(ingest_manager_t* mgr);
void ingest_print_streams(ingest_manager_t* mgr);
```

### Verification

```c
int verify_system(verify_report_t* report);
void verify_print_report(const verify_report_t* report);
int verify_is_healthy(const verify_report_t* report);
```

---

## 11. Files Created

- `core/include/ingest.h` - Ingestion API
- `core/include/verify.h` - Verification API
- `backend/cpu/ingest.c` - Ingestion implementation
- `backend/cpu/verify.c` - Verification implementation
- `io/sensors.json` - Sensor configuration
- `io/adapters/net_adapter.c` - HTTP/REST adapter
- `io/adapters/sim_adapter.c` - Simulation adapter

---

## 12. Next Steps

When live data stabilizes (coherence > 0.995 for ≥ 5 min):

→ **Phase 7: Distributed Swarm Nodes** (multi-machine Qallow links)

---

**Status**: ✅ Phase 6 Integrated into Qallow Unified Command System

