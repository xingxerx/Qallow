# Temporal Memory Architecture

## Overview

The Temporal Memory system is the foundational layer of the Qallow AGI architecture. It provides a three-level hierarchical memory system designed to:

1. **Track execution history** (episodic events)
2. **Learn semantic concepts** (embedding vectors)
3. **Detect performance drift** (coherence monitoring)

This architecture underpins all subsequent AGI capabilities: adaptive governance, stability monitoring, self-improvement protocols, and bias auditing.

---

## Architecture: Three-Level Memory Hierarchy

### Level 1: Episodic Buffer
**Purpose**: Track discrete execution events with context and scoring.

```c
typedef struct {
    uint64_t event_id;           // Unique identifier
    time_t timestamp;            // When it occurred
    char event_type[64];         // "execution", "decision", "error", etc.
    char event_description[256]; // Context and details
    float event_score;           // Quality/success score [0, 1]
} qallow_episodic_event_t;
```

**Capabilities**:
- Log events during execution phases
- Query events by type (filter)
- Retrieve most recent events (temporal ordering)
- Prune old events (memory management)
- TTL-based cleanup

**Use Cases**:
- Phase execution traces
- Decision checkpoints
- Error conditions
- Adaptive decision triggers

### Level 2: Semantic Embeddings
**Purpose**: Learn dense vector representations of concepts for similarity comparison.

```c
typedef struct {
    char concept_name[128];           // Concept identifier
    float embedding[128];              // 128-dimensional vector
    float confidence;                  // How reliable the embedding is [0, 1]
    int occurrence_count;              // How many times seen
} qallow_semantic_embedding_t;
```

**Capabilities**:
- Store concept vectors (pre-computed or learned)
- Retrieve embeddings by name
- Find semantically similar concepts (cosine similarity)
- Update embeddings (incremental learning)
- Track concept occurrence frequency

**Use Cases**:
- Map algorithm states to embedding space
- Find similar execution patterns
- Enable concept-based reasoning
- Support adaptive policy selection

### Level 3: Coherence Metrics
**Purpose**: Monitor system coherence and detect drift from baseline behavior.

```c
typedef struct {
    float temporal_gradient;      // Rate of change over time
    float baseline_deviation;     // Distance from established baseline
    float entropy_score;          // System entropy [0, 1]
    float harmony_score;          // Harmonic integration [0, 1]
    time_t last_baseline_update;  // When baseline was set
} qallow_coherence_metrics_t;
```

**Drift Detection Algorithm**:
```
drift = √[(entropy_diff)² + (harmony_diff)² + (gradient_diff)²] / √3

where:
  entropy_diff = current_entropy - baseline_entropy
  harmony_diff = current_harmony - baseline_harmony
  gradient_diff = current_gradient - baseline_gradient
```

**Capabilities**:
- Set baseline coherence metrics
- Update current metrics from phase data
- Calculate drift (L2 distance from baseline)
- Track drift history (circular buffer, 100 entries)
- Detect trends (increasing/decreasing/stable)
- Generate alerts when drift exceeds threshold

**Use Cases**:
- Detect when system deviates from normal operation
- Early warning before major failures
- Validate adaptive governance decisions
- Support self-improvement feedback loops

---

## API Overview

### Level 1: Episodic Events

```c
// Initialize memory system (max_events = 1000 by default)
qallow_temporal_memory_t mem = qallow_temporal_memory_init(max_events);

// Log an event
uint64_t event_id = qallow_temporal_memory_log_event(
    &mem,
    "execution",      // event_type
    "Phase 12 started", // description
    0.95f             // score [0, 1]
);

// Query events by type (NULL = all events)
qallow_episodic_event_t* events = qallow_temporal_memory_query_events(
    &mem,
    "execution",  // filter by type (or NULL for all)
    10            // max_results
);

// Prune old events (keep most recent N)
qallow_temporal_memory_prune_events(&mem, 500);

// Cleanup
qallow_temporal_memory_free(&mem);
```

### Level 2: Semantic Embeddings

```c
// Add or update an embedding
float embedding[128];
// ... populate embedding ...

int ret = qallow_temporal_memory_add_embedding(
    &mem,
    "stable_execution",  // concept_name
    embedding,           // 128-dim vector
    0.95f                // confidence
);

// Get specific embedding
qallow_semantic_embedding_t* emb = qallow_temporal_memory_get_embedding(
    &mem,
    "stable_execution"
);

// Find semantically similar concepts
qallow_semantic_embedding_t* similar = qallow_temporal_memory_nearest_embeddings(
    &mem,
    query_embedding,  // 128-dim query vector
    5                 // return top-5 most similar
);
```

### Level 3: Coherence & Drift

```c
// Set baseline coherence
qallow_temporal_memory_update_coherence(
    &mem,
    0.0f,   // temporal_gradient
    0.0f,   // baseline_deviation
    0.5f,   // entropy_score
    0.95f   // harmony_score
);
qallow_temporal_memory_set_baseline(&mem);

// Update current coherence (from phase metrics)
qallow_temporal_memory_update_coherence(
    &mem,
    0.05f,  // gradient changed
    0.02f,  // small deviation
    0.52f,  // slight entropy increase
    0.93f   // harmony decreased
);

// Detect drift
float drift = qallow_temporal_memory_detect_drift(&mem);
// drift is automatically recorded in history

// Get current coherence state
qallow_coherence_metrics_t coherence = qallow_temporal_memory_get_coherence(&mem);

// Set drift alert threshold
qallow_temporal_memory_set_drift_threshold(&mem, 0.15f);

// Analyze drift trends
float avg_drift = qallow_temporal_memory_get_average_drift(&mem, 20);
int trend = qallow_temporal_memory_get_drift_trend(&mem);
// trend: 1 = increasing, 0 = stable, -1 = decreasing
```

### Utilities

```c
// Export session data to JSON
qallow_temporal_memory_export_json(&mem, "session_audit.json");

// Print statistics to stdout
qallow_temporal_memory_print_stats(&mem);
```

---

## Integration Points

### Phase Runners
```c
// At phase start
qallow_temporal_memory_log_event(mem, "phase_start", "Phase 12", 1.0f);
qallow_temporal_memory_update_coherence(mem, 0, 0, baseline_entropy, baseline_harmony);
qallow_temporal_memory_set_baseline(mem);

// During phase execution
qallow_temporal_memory_log_event(mem, "execution", "Tick 5 complete", score);

// At phase end
qallow_temporal_memory_log_event(mem, "phase_complete", "Phase 12", final_score);
float drift = qallow_temporal_memory_detect_drift(mem);
if (drift > threshold) {
    fprintf(stderr, "ALERT: Coherence drift detected (%.3f)\n", drift);
}
```

### Adaptive Governance
```c
// Governance decisions use drift as input
if (drift > coherence_threshold) {
    // Apply corrective policy
    apply_stabilizing_policy(&algorithm, drift);
}
```

### Stability Monitors
```c
// Monitor uses temporal memory state
int trend = qallow_temporal_memory_get_drift_trend(mem);
if (trend > 0) {
    // Drift is increasing - take preventive action
    activate_stability_control();
}
```

### Self-Improvement
```c
// Self-improvement uses session data
qallow_temporal_memory_export_json(mem, "data/telemetry/session_audit.json");
// Post-session analysis can use this data to compute improvement metrics
```

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ Qallow Phase Execution                                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Phase Start                                                    │
│    ↓                                                            │
│  [log_event("phase_start", ...)]  ──→  Level 1 Buffer         │
│    ↓                                                            │
│  [update_coherence(baseline)]  ──────→  Level 3 Metrics       │
│  [set_baseline()]                                              │
│    ↓                                                            │
│  ┌─ Tick Loop ─────────────────────────────────────────┐      │
│  │                                                      │       │
│  │ Execute tick                                         │       │
│  │   ↓                                                  │       │
│  │ [log_event("execution", tick_score, ...)]           │       │
│  │   ↓                                                  │       │
│  │ [update_coherence(current_metrics)]                 │       │
│  │   ↓                                                  │       │
│  │ drift = [detect_drift()]  ──→  (stored + analyzed)  │       │
│  │   ↓                                                  │       │
│  │ if (drift > threshold): [alert]                     │       │
│  │                                                      │       │
│  │ (optional) [add_embedding(concept, vector)]         │       │
│  │                                                      │       │
│  └─────────────────────────────────────────────────────┘       │
│    ↓                                                            │
│  Phase End                                                      │
│    ↓                                                            │
│  [log_event("phase_complete", final_score)]                    │
│  [export_json("session_audit.json")]                           │
│                                                                 │
│ ┌─────────────────────────────────────────────────────────┐   │
│ │ Available for post-session analysis:                    │   │
│ │  - Event sequence (episodic)                            │   │
│ │  - Drift history (coherence trends)                     │   │
│ │  - Concept embeddings (semantic)                        │   │
│ │  - Performance metrics (effectiveness)                  │   │
│ └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Performance Characteristics

### Memory Usage
- **Episodic Buffer**: ~350 bytes/event (event_id, timestamp, type, description, score)
  - Default capacity: 1000 events = ~350 KB
  - Configurable: `qallow_temporal_memory_init(max_events)`
  
- **Embeddings**: ~520 bytes/embedding (name + 128 floats + metadata)
  - Default capacity: 100 embeddings = ~52 KB
  - Grows dynamically as needed
  
- **Coherence Metrics**: Fixed ~5 KB
  
- **Drift History**: Fixed ~400 bytes (100 float entries)

**Total Typical Footprint**: ~410 KB

### Computational Complexity
- **log_event()**: O(1) amortized
- **query_events()**: O(n) where n = total events
- **prune_events()**: O(1) with memmove
- **add_embedding()**: O(m) where m = total embeddings (linear search for duplicates)
- **nearest_embeddings()**: O(m*d + m log m) where d=128, m=embeddings (similarity + sort)
- **detect_drift()**: O(1)

### Similarity Search Optimization
Current implementation uses cosine similarity with bubble sort for ranking. For production deployments with >1000 embeddings, consider:
- KD-tree or ball-tree for O(log m) lookup
- SIMD-accelerated cosine similarity
- GPU-accelerated similarity matrix computation

---

## Usage Examples

### Example 1: Phase Execution Tracking

```c
// Initialize temporal memory
qallow_temporal_memory_t mem = qallow_temporal_memory_init(500);

// Phase 12 execution
qallow_temporal_memory_log_event(&mem, "phase_start", "Phase 12 elasticity", 1.0f);
qallow_temporal_memory_update_coherence(&mem, 0.0f, 0.0f, 0.55f, 0.92f);
qallow_temporal_memory_set_baseline(&mem);

for (int tick = 0; tick < 120; ++tick) {
    // Execute tick...
    
    // Log progress
    char desc[100];
    snprintf(desc, sizeof(desc), "Tick %d elasticity=%.3f", tick, elasticity);
    qallow_temporal_memory_log_event(&mem, "execution", desc, score);
    
    // Update coherence from current metrics
    qallow_temporal_memory_update_coherence(&mem, gradient, deviation, entropy, harmony);
    
    // Check for drift
    float drift = qallow_temporal_memory_detect_drift(&mem);
    if (drift > 0.2f) {
        fprintf(stderr, "[Phase 12] ⚠️ Drift Alert: %.3f\n", drift);
    }
}

// Log completion
qallow_temporal_memory_log_event(&mem, "phase_complete", "Phase 12 done", final_score);

// Export session data
qallow_temporal_memory_export_json(&mem, "data/telemetry/phase12_session.json");

// Print statistics
qallow_temporal_memory_print_stats(&mem);

qallow_temporal_memory_free(&mem);
```

### Example 2: Concept Learning & Similarity

```c
// Initialize with embeddings for learned concepts
qallow_temporal_memory_t mem = qallow_temporal_memory_init(1000);

// Learn concept: "stable_execution"
float stable_vec[128] = {...};  // Pre-computed or learned
qallow_temporal_memory_add_embedding(&mem, "stable_execution", stable_vec, 0.95f);

// Learn concept: "phase_transition"
float transition_vec[128] = {...};
qallow_temporal_memory_add_embedding(&mem, "phase_transition", transition_vec, 0.85f);

// Learn concept: "error_condition"
float error_vec[128] = {...};
qallow_temporal_memory_add_embedding(&mem, "error_condition", error_vec, 0.70f);

// During adaptive governance: find most similar concept
float current_state_vec[128] = {generate from current metrics};
qallow_semantic_embedding_t* similar = qallow_temporal_memory_nearest_embeddings(
    &mem, current_state_vec, 1
);

printf("Current state is most similar to: %s (%.3f similarity)\n",
       similar[0].concept_name, similar[0].confidence);

// Use similarity to select policy
if (strcmp(similar[0].concept_name, "stable_execution") == 0) {
    apply_standard_policy();
} else if (strcmp(similar[0].concept_name, "phase_transition") == 0) {
    apply_transition_policy();
}

free(similar);
qallow_temporal_memory_free(&mem);
```

---

## Integration with Other Subsystems

### Adaptive Governance (Phase 16)
- Reads: Drift metrics, coherence state, concept embeddings
- Writes: Policy decisions logged as events
- Feedback: Uses drift as control signal

### Stability Monitors (Phase 17)
- Reads: Drift history, trend analysis, coherence gradient
- Writes: Stability alerts as events
- Feedback: Validates coherence bounds

### Self-Improvement Protocol (Phase 18)
- Reads: Full session data (JSON export)
- Writes: Improvement metrics and feedback
- Feedback: Updates baselines for next session

### Bias Auditing (Phase 19)
- Reads: Event sequence, embedding drift, coherence history
- Writes: Bias reports and audit logs
- Feedback: Triggers corrective policies

---

## Testing

Run the comprehensive test suite:

```bash
cd /home/xing/Qallow/build
ctest -R test_temporal_memory --output-on-failure
```

Test coverage includes:
- ✅ Event logging and querying (7 tests)
- ✅ Embedding addition and similarity (5 tests)
- ✅ Coherence tracking and drift (5 tests)
- ✅ Drift statistics and trends (3 tests)
- ✅ JSON export (1 test)
- ✅ Statistics output (1 test)
- ✅ Full integration lifecycle (1 test)

---

## Future Enhancements

1. **GPU-Accelerated Similarity Search**
   - CUDA kernels for cosine similarity computation
   - Batch similarity queries for efficiency

2. **Learned Embedding Generation**
   - Integration with neural components for dynamic embedding learning
   - Support for multi-modal embeddings

3. **Advanced Time Series Analysis**
   - ARIMA for drift prediction
   - Anomaly detection using isolation forests
   - Seasonality analysis for cyclic patterns

4. **Distributed Memory**
   - Multi-GPU memory synchronization
   - Federated learning across nodes
   - Sharded embedding storage

5. **Memory Compression**
   - Event summarization (abstract old events)
   - Embedding quantization (reduce precision)
   - Automatic memory optimization

---

## References

- **Files**: `include/qallow/temporal_memory.h`, `src/runtime/temporal_memory.c`, `tests/test_temporal_memory.c`
- **Build**: `cmake --build build --target qallow_test_temporal_memory`
- **Architecture**: See `docs/ARCHITECTURE_SPEC.md` for full AGI framework
