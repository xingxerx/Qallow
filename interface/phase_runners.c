#include "phase_runners.h"
#include "qallow/logging.h"
#include "qallow_phase12.h"
#include "qallow_phase13.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Real implementation for qallow_phase12_runner
int qallow_phase12_runner(int argc, char** argv) {
    qallow_log_info("BENCHMARK", "Calling real: qallow_phase12_runner");
    const char* audit_tag = "benchmark_p12";
    int ticks = 1000;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--audit-tag") == 0 && i + 1 < argc) {
            audit_tag = argv[++i];
        } else if (strcmp(argv[i], "--ticks") == 0 && i + 1 < argc) {
            ticks = atoi(argv[++i]);
        }
    }
    
    return run_phase12_elasticity(audit_tag, NULL, ticks, 0.1f);
}

// Real implementation for qallow_phase13_runner
int qallow_phase13_runner(int argc, char** argv) {
    qallow_log_info("BENCHMARK", "Calling real: qallow_phase13_runner");
    const char* audit_tag = "benchmark_p13";
    int ticks = 2000;
    int pockets = 128;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--audit-tag") == 0 && i + 1 < argc) {
            audit_tag = argv[++i];
        } else if (strcmp(argv[i], "--ticks") == 0 && i + 1 < argc) {
            ticks = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--pockets") == 0 && i + 1 < argc) {
            pockets = atoi(argv[++i]);
        }
    }

    return run_phase13_harmonic(audit_tag, NULL, pockets, ticks, 0.5f);
}
