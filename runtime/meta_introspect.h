#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct learn_event_s {
    const char* phase;
    const char* module;
    const char* objective_id;
    float duration_s;
    float coherence;
    float ethics;
} learn_event_t;

// Sequential introspection trigger (Phase 16 enhancement)
typedef struct {
    int trigger_id;
    long timestamp_ms;
    const char* trigger_type;  // "coherence_drop", "ethics_violation", "latency_spike"
    float metric_value;
    float threshold;
    int severity;  // 0=low, 1=medium, 2=high
} introspection_trigger_t;

// Sequential introspection result
typedef struct {
    int trigger_id;
    float introspection_score;
    const char* recommendation;
    int confidence;  // 0-100
} introspection_result_t;

int meta_introspect_configure(const char* base_dir, const char* objective_map_path);
void meta_introspect_enable(int enabled);
int meta_introspect_enabled(void);
void meta_introspect_set_gpu_available(int available);
void meta_introspect_push(const learn_event_t* event);
void meta_introspect_flush(void);
void meta_introspect_apply_environment_defaults(void);
const char* meta_introspect_log_dir(void);
int meta_introspect_export_pocket_map(const char* output_path);

// Sequential introspection functions (Phase 16 stabilization)
int meta_introspect_log_trigger(const introspection_trigger_t* trigger,
                                const char* log_path);
int meta_introspect_sequential_reasoning(const introspection_trigger_t* trigger,
                                         introspection_result_t* result,
                                         const char* log_path);

int qallow_meta_introspect_gpu(const float* durations,
                               const float* coherence,
                               const float* ethics,
                               float* improvement_scores,
                               int count);

#ifdef __cplusplus
}
#endif
