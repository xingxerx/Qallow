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

int meta_introspect_configure(const char* base_dir, const char* objective_map_path);
void meta_introspect_enable(int enabled);
int meta_introspect_enabled(void);
void meta_introspect_set_gpu_available(int available);
void meta_introspect_push(const learn_event_t* event);
void meta_introspect_flush(void);
void meta_introspect_apply_environment_defaults(void);
const char* meta_introspect_log_dir(void);
int meta_introspect_export_pocket_map(const char* output_path);

#ifdef __cplusplus
}
#endif
