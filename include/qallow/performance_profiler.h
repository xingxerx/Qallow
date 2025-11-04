#ifndef QALLOW_PERFORMANCE_PROFILER_H
#define QALLOW_PERFORMANCE_PROFILER_H

#include <time.h>
#include <stdint.h>

/**
 * @file performance_profiler.h
 * @brief Performance profiling instrumentation for phase execution tracking
 * 
 * Provides fine-grained timing data collection for:
 * - Individual phase execution times
 * - CPU vs GPU execution comparison
 * - Memory allocation patterns
 * - Bottleneck identification
 * 
 * Usage:
 *   qallow_profiler_t prof = qallow_profiler_init();
 *   qallow_profiler_mark(&prof, "phase13_start");
 *   // ... phase work ...
 *   qallow_profiler_mark(&prof, "phase13_end");
 *   qallow_profiler_export_csv(&prof, "profile.csv");
 */

typedef struct {
    char label[128];
    clock_t clock_time;
    int64_t usec_timestamp;
    double elapsed_ms;  /* elapsed since start */
} qallow_profile_mark_t;

typedef struct {
    qallow_profile_mark_t marks[256];
    int mark_count;
    clock_t start_clock;
    int64_t start_usec;
    
    /* Aggregated metrics */
    double phase12_time_ms;
    double phase13_time_ms;
    double phase14_time_ms;
    double phase15_time_ms;
    double gpu_vs_cpu_ratio;  /* GPU time / CPU time */
} qallow_profiler_t;

/**
 * Initialize profiler (resets all marks)
 */
qallow_profiler_t qallow_profiler_init(void);

/**
 * Record a timing mark with label
 * @param prof - profiler instance

 */
void qallow_profiler_mark(qallow_profiler_t* prof, const char* label);

/**
 * Record phase execution time
 * @param prof - profiler instance
 * @param phase_num - phase number (12-15)
 * @param elapsed_ms - execution time in milliseconds
 */
void qallow_profiler_record_phase(qallow_profiler_t* prof, 
                                   int phase_num, 
                                   double elapsed_ms);

/**
 * Record GPU vs CPU ratio for comparison
 * @param prof - profiler instance
 * @param gpu_ms - GPU execution time
 * @param cpu_ms - CPU execution time (fallback)
 */
void qallow_profiler_record_gpu_cpu_ratio(qallow_profiler_t* prof,
                                          double gpu_ms,
                                          double cpu_ms);

/**
 * Export profiling data to CSV
 * @param prof - profiler instance
 * @param csv_path - output CSV file path
 * @return 0 on success, -1 on error
 */
int qallow_profiler_export_csv(const qallow_profiler_t* prof, 
                                const char* csv_path);

/**
 * Export profiling data to JSON
 * @param prof - profiler instance
 * @param json_path - output JSON file path
 * @return 0 on success, -1 on error
 */
int qallow_profiler_export_json(const qallow_profiler_t* prof,
                                 const char* json_path);

/**
 * Print profiling summary to stdout
 * @param prof - profiler instance
 */
void qallow_profiler_print_summary(const qallow_profiler_t* prof);

/**
 * Get elapsed time since profiler initialization (ms)
 */
double qallow_profiler_elapsed_ms(const qallow_profiler_t* prof);

#endif  /* QALLOW_PERFORMANCE_PROFILER_H */
