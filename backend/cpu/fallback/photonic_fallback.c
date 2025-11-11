// Fallback for backend/cuda/photonic.cu
#include "qallow/runtime.h"

void photonic_simulation_fallback(qallow_state_t* state) {
    // CPU implementation or empty stub
    qallow_log(state, LOG_LEVEL_WARN, "CUDA not available. Photonic simulation running in fallback mode.");
}
