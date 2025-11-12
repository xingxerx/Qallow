#include "qallow/logging.h"
#include "qallow/profiling.h"
#include "phase_runners.h"
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

    char phase12_prog[] = "qallow_examples";
    char phase12_cmd[] = "phase12";
    char phase12_ticks[] = "--ticks=200";
    char phase12_eps[] = "--eps=0.0005";
    char* phase12_args[] = {phase12_prog, phase12_cmd, phase12_ticks, phase12_eps};
    if (run_phase("phase12", qallow_phase12_runner, 4, phase12_args) != 0) {
        qallow_log_error("benchmark.phase12.failed", "ticks=%d", 200);
        return EXIT_FAILURE;
    }

    char phase13_prog[] = "qallow_examples";
    char phase13_cmd[] = "phase13";
    char phase13_nodes[] = "--nodes=16";
    char phase13_ticks[] = "--ticks=400";
    char phase13_k[] = "--k=0.001";
    char* phase13_args[] = {phase13_prog, phase13_cmd, phase13_nodes, phase13_ticks, phase13_k};
    if (run_phase("phase13", qallow_phase13_runner, 5, phase13_args) != 0) {
        qallow_log_error("benchmark.phase13.failed", "nodes=%d ticks=%d", 16, 400);
        return EXIT_FAILURE;
    }

    qallow_log_info("benchmark.complete", "phases=%d", 2);
    qallow_logging_flush();
    return EXIT_SUCCESS;
}
