// Fallback for backend/cuda/phase16_meta_introspect.cu
#include "qallow/runtime.h"

void phase16_meta_introspect_fallback(qallow_state_t* state) {
    // CPU implementation or empty stub
    qallow_log(state, LOG_LEVEL_WARN, "CUDA not available. Phase 16 (Meta-Introspection) running in fallback mode.");
}
