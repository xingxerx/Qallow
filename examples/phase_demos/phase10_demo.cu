#include "phase_demo_common.cuh"
#include "logging.h"
#include "profiling.h"

#include <cstdlib>

int main(int argc, char** argv) {
    int ticks = 128;
    if (argc > 1) {
        ticks = std::atoi(argv[1]);
    }

    qallow_logging_init();
    QALLOW_PROFILE_SCOPE("phase10_demo");
    int rc = run_phase_demo_kernel(10, ticks);
    if (rc != 0) {
        qallow_log_error("examples.phase10.fail", "code", rc);
        return rc;
    }

    qallow_log_info("examples.phase10.ok", "ticks", ticks);
    qallow_logging_flush();
    return 0;
}
