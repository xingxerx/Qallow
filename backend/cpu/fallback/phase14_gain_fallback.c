// Fallback for backend/cuda/phase14_gain.cu
#include "qallow/runtime.h"

void phase14_gain_fallback(qallow_state_t* state) {
    // CPU implementation or empty stub
    qallow_log(state, LOG_LEVEL_WARN, "CUDA not available. Phase 14 (Gain) running in fallback mode.");
}
