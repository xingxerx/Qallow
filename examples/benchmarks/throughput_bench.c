#include "qallow/logging.h"
#include "qallow/profiling.h"
#include "qallow_phase12.h"
#include "qallow_phase13.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double wall_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static int run_phase(const char* phase, int (*runner)(int, char**), int argc, char** argv) {
    double start = wall_ms();
    int rc = 0;
    QALLOW_PROFILE_SCOPE(phase) {
        rc = runner(argc, argv);
    }
    double elapsed = wall_ms() - start;

    qallow_log_info("benchmark.phase", "phase=%s elapsed_ms=%.3f", phase, elapsed);
    return rc;
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    qallow_logging_init();

    const char* phase12_args[] = {"qallow_examples", "phase12", "--ticks=200", "--eps=0.0005"};
    if (run_phase("phase12", qallow_phase12_runner, 4, (char**)phase12_args) != 0) {
        qallow_log_error("benchmark.phase12.failed", "ticks=%d", 200);
        return EXIT_FAILURE;
    }

    const char* phase13_args[] = {"qallow_examples", "phase13", "--nodes=16", "--ticks=400", "--k=0.001"};
    if (run_phase("phase13", qallow_phase13_runner, 5, (char**)phase13_args) != 0) {
        qallow_log_error("benchmark.phase13.failed", "nodes=%d ticks=%d", 16, 400);
        return EXIT_FAILURE;
    }

    qallow_log_info("benchmark.complete", "phases=%d", 2);
    qallow_logging_flush();
    return EXIT_SUCCESS;
}
