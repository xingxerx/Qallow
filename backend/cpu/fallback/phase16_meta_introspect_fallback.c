// Fallback for backend/cuda/phase16_meta_introspect.cu
#include "qallow/logging.h"

void phase16_meta_introspect_fallback(void) {
    // CPU implementation or empty stub
    qallow_log_warn("phase16_meta_introspect_fallback", "CUDA not available. Phase 16 (Meta-Introspection) running in fallback mode.");
}
