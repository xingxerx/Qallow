// Fallback for backend/cuda/quantum.cu
#include "qallow/logging.h"

void quantum_simulation_fallback(void) {
    // CPU implementation or empty stub
    qallow_log_warn("quantum_fallback", "CUDA not available. Quantum simulation running in fallback mode.");
}
