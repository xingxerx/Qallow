# Performance Profiling Sections - Enhancement Guide

**Status:** Documentation updated for multi-launch workload optimization  
**Last Updated:** November 4, 2025

---

## Overview

The Nsight Compute profiling sections provide detailed GPU performance analysis. This guide clarifies TODO items and optimization strategies for multi-launch workloads.

---

## Section-by-Section Enhancement

### 1. Issue Slot Utilization (`sections/IssueSlotUtilization.py`)

**TODO Clarification:**
```python
# TODO: For multi-launch workloads, this is simply the average of the maximum warps
# over all launches. Instead, it should be the weighed average, where the weight
# would be given by the relative duration of the launch.
```

**What This Means:**
- **Current:** Takes raw average of warp counts across launches
- **Desired:** Weight each launch by its execution duration
- **Impact:** More accurate utilization metrics for workloads with variable launch times

**Implementation Strategy:**
```python
# Multi-launch weighted average calculation:
# weighted_avg = Σ(warp_count_i * duration_i) / Σ(duration_i)
# Where duration_i is the execution time of each launch

def get_weighted_theoretical_warps(metrics_per_launch):
    """Calculate weighted average across launches."""
    total_duration = sum(m.duration_ms for m in metrics_per_launch)
    if total_duration <= 0:
        return 0
    
    weighted_sum = sum(
        m.theoretical_warps * m.duration_ms 
        for m in metrics_per_launch
    )
    return weighted_sum / total_duration
```

**Optimization Target:** ≥ 0.6 issue active rate

---

### 2. Launch Statistics (`sections/LaunchStatistics.py`)

**TODO Clarification:**
```python
# TODO: We could get a global estimate by weighing the speedup by the
# relative importance/duration of each launch.
```

**What This Means:**
- **Current:** Simple average of speedup potential across launches
- **Desired:** Weight speedup by launch duration/importance
- **Impact:** Better prioritization of optimization targets

**Example:**
```
Launch A: 10ms, 20% speedup potential
Launch B: 90ms, 5% speedup potential

Simple avg: (20% + 5%) / 2 = 12.5%
Weighted:   (20%*10 + 5%*90) / (10+90) = 6.5%  ← More realistic!
```

---

### 3. Requested Metrics (`sections/RequestedMetrics.py`)

**TODO Clarification:**
```python
# TODO: switch to enum.Enum once this is available in static interpreter
```

**What This Means:**
- **Current:** Using dictionaries for metric requests (non-type-safe)
- **Desired:** Use Python Enum for type safety and IDE support
- **Impact:** Fewer runtime errors, better code completion

**Implementation (when available):**
```python
from enum import Enum

class MetricType(Enum):
    """Type-safe metric definitions."""
    ACTIVE_WARPS = "smsp__warps_active.avg.per_cycle_active"
    ELIGIBLE_WARPS = "smsp__warps_eligible.avg.per_cycle_active"
    MEMORY_THROUGHPUT = "dram__bytes_read.sum + dram__bytes_written.sum"
    COMPUTE_UTILIZATION = "sm__pipe_fma_active.sum / sm__cycles_active"

# Usage:
request = MetricRequest(MetricType.ACTIVE_WARPS)
```

---

### 4. Theoretical Occupancy (`sections/TheoreticalOccupancy.py`)

**TODO Clarification:**
```python
# TODO: We could get a global estimate by forming the weighted average of all
# occupancy values, weighted by the duration/importance of each launch.
```

**What This Means:**
- Calculate occupancy for each kernel launch
- Weight by execution duration
- Provides representative occupancy metric

**Example:**
```
Launch 1: 50% occupancy × 5ms = 250
Launch 2: 75% occupancy × 15ms = 1125
Global occupancy = (250 + 1125) / 20ms = 68.75%
```

**Optimization Targets:**
- Target occupancy: 80%+
- Minimum occupancy: 50%
- Critical: < 40% indicates severe underutilization

---

## Multi-Launch Workload Optimization Pattern

### Understanding Multi-Launch Workloads

Multi-launch workloads run multiple kernel invocations with different:
- Grid dimensions
- Block dimensions  
- Register usage
- Shared memory usage
- Execution time

### Profiling Multi-Launch Workloads

**Step 1: Identify Each Launch**
```bash
# Capture with Nsight Compute
ncu --set full app
```

**Step 2: Extract Per-Launch Metrics**
```python
launches = extract_launches_from_report()
for i, launch in enumerate(launches):
    print(f"Launch {i}:")
    print(f"  Grid:       {launch.grid_dim}")
    print(f"  Occupancy:  {launch.occupancy:.1%}")
    print(f"  Duration:   {launch.duration_ms:.2f}ms")
    print(f"  Throughput: {launch.throughput_gbps:.1f} GB/s")
```

**Step 3: Weight by Duration**
```python
metrics = ['occupancy', 'throughput', 'utilization']

for metric in metrics:
    total_duration = sum(l.duration_ms for l in launches)
    weighted = sum(
        l.__dict__[metric] * l.duration_ms 
        for l in launches
    ) / total_duration
    print(f"{metric}: {weighted:.2f} (weighted)")
```

**Step 4: Optimize High-Impact Launches**
- Focus on launches with:
  - Longest duration
  - Lowest efficiency
  - Highest potential speedup

---

## Common Optimization Opportunities

### Issue Slot Utilization < 0.6

**Diagnosis:**
- Check warp count (may be too low)
- Check for stalls (memory, dependency)
- Check instruction diversity

**Fix:**
```c
// Add more parallel work
// Increase block size (if register-limited)
// Use prefetching for memory loads
// Reduce branch divergence
```

### Low Occupancy (< 50%)

**Diagnosis:**
- Register pressure too high
- Not enough threads per block
- Shared memory pressure

**Fix:**
```cuda
// Option 1: Reduce registers per thread
__global__ void kernel(...) {
    // Use local arrays instead of multiple registers
    // Share computation results
}

// Option 2: Increase threads per block
// kernel<<<grid, 256>>>  // from 128
```

### High Memory Stall Rate

**Diagnosis:**
- Cache misses
- Uncoalesced memory access
- Memory bandwidth bottleneck

**Fix:**
```cuda
// Improve data layout
// Coalesce memory access patterns
// Use shared memory cache
// Reduce memory footprint
```

---

## Performance Targets

### Qallow Phase Execution

| Phase | Target Occupancy | Target Issue Active | Target Time |
|-------|-----------------|-------------------|------------|
| Phase 12 | 80% | 0.7+ | < 10ms |
| Phase 13 | 75% | 0.65+ | < 15ms |
| Phase 14 | 70% | 0.6+ | < 20ms |
| Phase 15 | 65% | 0.55+ | < 25ms |

---

## Tools & Resources

### Profiling Tools
- **Nsight Compute:** GUI profiler with detailed metrics
- **ncu:** Command-line profiler
- **nvprof:** Legacy profiler (still works)
- **Nsight Systems:** Timeline profiler

### Analysis Tools
```bash
# Generate profiling report
python3 scripts/telemetry_dashboard.py data/logs/phase13.csv

# Compare phases
python3 scripts/telemetry_dashboard.py --compare phase12.csv phase13.csv

# Analyze GPU benchmark
python3 scripts/telemetry_dashboard.py --gpu-bench gpu_profile.json
```

### Documentation
- [NVIDIA Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Qallow Performance Analysis](../PERFORMANCE_ANALYSIS.md) *(to be created)*

---

## Implementation Checklist

- [ ] Add duration tracking to all kernel launches
- [ ] Implement weighted average calculations
- [ ] Create per-launch profiling reports
- [ ] Add multi-launch optimization guide
- [ ] Benchmark before/after optimization
- [ ] Document findings in telemetry
- [ ] Update phase performance baselines

---

## Next Steps

1. **Week 1:** Enhance issue slot utilization with weighted averages
2. **Week 2:** Add multi-launch analysis tools
3. **Week 3:** Create optimization recommendations engine
4. **Week 4:** Integrate with CI/CD dashboard

**Owner:** Performance team  
**Priority:** Medium (nice-to-have enhancement)
