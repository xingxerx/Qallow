// Fallback for backend/cuda/quantum.cu
#include "qallow/runtime.h"

void quantum_simulation_fallback(qallow_state_t* state) {
    // CPU implementation or empty stub
    qallow_log(state, LOG_LEVEL_WARN, "CUDA not available. Quantum simulation running in fallback mode.");
}
