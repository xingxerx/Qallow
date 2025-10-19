#include "phase_demo_common.cuh"
#include "qallow/logging.h"
#include "qallow/profiling.h"

#include <cstdlib>

int main(int argc, char** argv) {
    int ticks = 128;
    if (argc > 1) {
        ticks = std::atoi(argv[1]);
    }

    qallow_logging_init();
    int rc = 0;
    QALLOW_PROFILE_SCOPE("phase02_demo") {
        rc = run_phase_demo_kernel(2, ticks);
    }
    if (rc != 0) {
        qallow_log_error("examples.phase02.fail", "code=%d", rc);
        return rc;
    }

    qallow_log_info("examples.phase02.ok", "ticks=%d", ticks);
    qallow_logging_flush();
    return 0;
}
