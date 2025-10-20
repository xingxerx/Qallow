#pragma once

#ifdef __cplusplus
extern "C" {
#endif

int qallow_meta_introspect_gpu(const float* durations,
                               const float* coherence,
                               const float* ethics,
                               float* improvement_scores,
                               int count);

#ifdef __cplusplus
}
#endif
