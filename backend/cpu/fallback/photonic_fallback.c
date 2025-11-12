// Fallback for backend/cuda/photonic.cu
#include "qallow/logging.h"

void photonic_simulation_fallback(void) {
    // CPU implementation or empty stub
    qallow_log_warn("photonic_fallback", "CUDA not available. Photonic simulation running in fallback mode.");
}
