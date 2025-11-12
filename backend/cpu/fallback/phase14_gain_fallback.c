// Fallback for backend/cuda/phase14_gain.cu
#include "qallow/logging.h"

void phase14_gain_fallback(void) {
    // CPU implementation or empty stub
    qallow_log_warn("phase14_gain_fallback", "CUDA not available. Phase 14 (Gain) running in fallback mode.");
}
