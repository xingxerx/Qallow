# Sequential Thinking Integration Guide

Quick reference for integrating sequential thinking enhancements into Qallow's runtime.

## 1. Ethics Pipeline Integration (Phases 8-10)

### Include Header
```c
#include "ethics_core.h"
```

### Usage Example
```c
// After computing ethics score
ethics_model_t model;
ethics_metrics_t metrics;
ethics_model_load(&model, NULL, NULL);

// ... populate metrics ...

// Trace the decision sequence
const char* log_path = "data/logs/ethics_trace.csv";
ethics_trace_decision_sequence(&model, &metrics, log_path);
```

### Output
- **File**: `data/logs/ethics_trace.csv`
- **Format**: CSV with columns: step_id, timestamp_ms, rule_name, input_value, threshold, verdict, intervention_type
- **Rows**: 5 per decision (safety, clarity, human, reality_drift, total_score)

### Integration Points
- Phase 8 (Ethics): After `ethics_score_core()`
- Phase 9 (Ethics Reasoning): After verdict computation
- Phase 10 (Ethics Feedback): After feedback application

---

## 2. Meta-Introspection Integration (Phase 16)

### Include Header
```c
#include "meta_introspect.h"
```

### Usage Example
```c
// When a trigger is detected
introspection_trigger_t trigger = {
    .trigger_id = 1,
    .timestamp_ms = get_current_time_ms(),
    .trigger_type = "coherence_drop",
    .metric_value = current_coherence,
    .threshold = coherence_threshold,
    .severity = 1  // 0=low, 1=medium, 2=high
};

introspection_result_t result;
const char* log_path = "data/logs/introspection_trace.csv";
meta_introspect_sequential_reasoning(&trigger, &result, log_path);

// Use result
printf("Recommendation: %s (confidence: %d%%)\n", 
       result.recommendation, result.confidence);
```

### Output
- **File**: `data/logs/introspection_trace.csv`
- **Format**: CSV + comment lines with results
- **Trigger Types**: coherence_drop, ethics_violation, latency_spike

### Integration Points
- Phase 16 (Meta-Introspection): On trigger detection
- Phase 15 (Coherence): When coherence drops below threshold
- Phase 14 (Gain): When latency spikes detected

---

## 3. Sequential Benchmarking

### Run Benchmark
```bash
bash /root/Qallow/tests/sequential_phase_benchmark.sh
```

### Output
- **File**: `data/logs/sequential_benchmark.csv`
- **Format**: CSV with columns: phase_id, phase_name, latency_ms, coherence_score, memory_mb, status
- **Includes**: Performance insights and recommendations

### Integration into CI/CD
```bash
# Add to build pipeline
./tests/sequential_phase_benchmark.sh
if [ $? -eq 0 ]; then
    echo "✓ Sequential benchmark passed"
else
    echo "✗ Sequential benchmark failed"
    exit 1
fi
```

---

## 4. Building with Sequential Enhancements

### CMake Integration
```cmake
# In CMakeLists.txt
add_executable(test_ethics_sequential 
    tests/unit/test_ethics_sequential.c
    algorithms/ethics_core.c
)

add_executable(test_meta_introspect_sequential
    tests/unit/test_meta_introspect_sequential.c
    runtime/meta_introspect.c
)
```

### Build Commands
```bash
cd /root/Qallow/build
cmake ..
make -j$(nproc)

# Run tests
./test_ethics_sequential
./test_meta_introspect_sequential
```

---

## 5. Monitoring & Analysis

### View Ethics Audit Trail
```bash
tail -f data/logs/ethics_trace.csv
```

### View Introspection Trace
```bash
tail -f data/logs/introspection_trace.csv
```

### Analyze Benchmark Results
```bash
# Show slowest phases
sort -t',' -k3 -rn data/logs/sequential_benchmark.csv | head -5

# Show average latency
awk -F',' 'NR>1 {sum+=$3; count++} END {print "Avg latency: " sum/count "ms"}' \
    data/logs/sequential_benchmark.csv
```

---

## 6. Configuration

### Log Paths
- Ethics: `data/logs/ethics_trace.csv`
- Introspection: `data/logs/introspection_trace.csv`
- Benchmark: `data/logs/sequential_benchmark.csv`

### Thresholds (Configurable)
```c
// Ethics thresholds (in config/thresholds.json)
{
    "min_safety": 0.7,
    "min_clarity": 0.65,
    "min_human": 0.6,
    "min_total": 1.85,
    "max_reality_drift": 0.25
}

// Introspection severity levels
// 0 = low, 1 = medium, 2 = high
```

---

## 7. Troubleshooting

### Ethics Trace Not Generated
```bash
# Check if log directory exists
mkdir -p data/logs

# Verify ethics_core.c is compiled
grep "ethics_trace_decision_sequence" build/CMakeFiles/*.dir/link.txt

# Check file permissions
ls -la data/logs/
```

### Introspection Trace Not Generated
```bash
# Verify meta_introspect.c is compiled
grep "meta_introspect_sequential_reasoning" build/CMakeFiles/*.dir/link.txt

# Check trigger detection is working
grep "trigger_type" data/logs/introspection_trace.csv
```

### Benchmark Script Issues
```bash
# Make script executable
chmod +x tests/sequential_phase_benchmark.sh

# Run with verbose output
bash -x tests/sequential_phase_benchmark.sh

# Check phase binaries exist
ls -la build/phase*_demo
```

---

## 8. Performance Tuning

### Optimize Ethics Logging
- Batch writes to reduce I/O overhead
- Use buffered file operations
- Consider async logging for high-frequency calls

### Optimize Introspection Reasoning
- Cache trigger analysis results
- Use lookup tables for severity adjustments
- Profile with `perf` or `nsys`

### Benchmark Optimization
- Run phases in parallel (if safe)
- Use profiling tools: `nsys profile`, `ncu`
- Compare against Heron baseline (150,000 CLOPS)

---

## 9. Example: Complete Integration

```c
// In your phase execution code
#include "ethics_core.h"
#include "meta_introspect.h"

void execute_phase_with_sequential_thinking(void) {
    // Phase 8-10: Ethics with sequential logging
    ethics_model_t model;
    ethics_metrics_t metrics;
    
    ethics_model_load(&model, NULL, NULL);
    // ... populate metrics ...
    
    // Log sequential decision
    ethics_trace_decision_sequence(&model, &metrics, 
                                   "data/logs/ethics_trace.csv");
    
    // Phase 16: Meta-introspection with sequential reasoning
    if (coherence < threshold) {
        introspection_trigger_t trigger = {
            .trigger_id = event_id++,
            .timestamp_ms = get_time_ms(),
            .trigger_type = "coherence_drop",
            .metric_value = coherence,
            .threshold = threshold,
            .severity = 1
        };
        
        introspection_result_t result;
        meta_introspect_sequential_reasoning(&trigger, &result,
                                            "data/logs/introspection_trace.csv");
        
        // Apply recommendation
        apply_recommendation(result.recommendation);
    }
}
```

---

## 10. References

- **Implementation**: `SEQUENTIAL_THINKING_IMPLEMENTATION.md`
- **Ethics Core**: `core/include/ethics_core.h`
- **Meta-Introspect**: `runtime/meta_introspect.h`
- **Tests**: `tests/unit/test_ethics_sequential.c`, `test_meta_introspect_sequential.c`
- **Benchmark**: `tests/sequential_phase_benchmark.sh`

---

**Last Updated**: 2025-10-24

