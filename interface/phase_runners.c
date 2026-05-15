#include "phase_runners.h"
#include "qallow/logging.h"
#include "qallow_phase11.h"
#include "qallow_phase1.h"
#include "qallow_phase2.h"
#include "qallow_phase3.h"
#include "qallow_phase4.h"
#include "qallow_phase16.h"
#include "qallow_phase17.h"
#include "qallow_phase18.h"
#include "qallow_phase19.h"
#include "qallow_phase20.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

/* Forward declarations for external phase binaries */
static int execute_phase_binary(int phase_num, int argc, char** argv);

/* Note: Phase 11, 14-15 runners are implemented in interface/main.c */

/* Phase 12: Elasticity Simulation */
int qallow_phase1_runner(int argc, char** argv) {
    qallow_log_info("BENCHMARK", "Calling real: qallow_phase1_runner");
    const char* audit_tag = "benchmark_p12";
    int ticks = 1000;
    float eps = 0.1f;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--audit-tag") == 0 && i + 1 < argc) {
            audit_tag = argv[++i];
        } else if (strcmp(argv[i], "--ticks") == 0 && i + 1 < argc) {
            ticks = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--eps") == 0 && i + 1 < argc) {
            eps = (float)atof(argv[++i]);
            if (eps < 0.0f) eps = 0.0f;
        }
    }

    return run_phase1_elasticity(audit_tag, NULL, ticks, eps);
}

/* Phase 13: Harmonic Propagation */
int qallow_phase2_runner(int argc, char** argv) {
    qallow_log_info("BENCHMARK", "Calling real: qallow_phase2_runner");
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

    return run_phase2_harmonic(audit_tag, NULL, pockets, ticks, 0.5f);
}

/* Helper function to execute external phase binaries */
static int execute_phase_binary(int phase_num, int argc, char** argv) {
    const char* phase_paths[] = {
        NULL,  // 0
        NULL,  // 1-15 handled by other runners
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        "phases/phase_16_constraint_validation",  // 16
        "phases/phase_17_memory",                 // 17
        "phases/phase_18_multiplayer",            // 18
        "phases/phase_19_audit",                  // 19
        "phases/phase_20_result_synthesis"        // 20
    };

    if (phase_num < 16 || phase_num > 20) {
        fprintf(stderr, "[PHASE%d] ERROR: Invalid phase number\n", phase_num);
        return 1;
    }

    const char* phase_path = phase_paths[phase_num];

    /* Check if phase binary exists */
    if (access(phase_path, X_OK) != 0) {
        fprintf(stderr, "[PHASE%d] ERROR: Phase binary not found at %s\n", phase_num, phase_path);
        return 1;
    }

    /* Fork and execute the phase binary */
    pid_t pid = fork();
    if (pid < 0) {
        fprintf(stderr, "[PHASE%d] ERROR: Failed to fork process\n", phase_num);
        return 1;
    }

    if (pid == 0) {
        /* Child process: build argument array and execute */
        char* phase_argv[argc + 1];
        phase_argv[0] = (char*)phase_path;
        for (int i = 1; i < argc; i++) {
            phase_argv[i] = argv[i];
        }
        phase_argv[argc] = NULL;

        execv(phase_path, phase_argv);

        /* If execv returns, it failed */
        fprintf(stderr, "[PHASE%d] ERROR: Failed to execute phase\n", phase_num);
        exit(1);
    } else {
        /* Parent process: wait for child */
        int status;
        waitpid(pid, &status, 0);

        if (WIFEXITED(status)) {
            return WEXITSTATUS(status);
        } else {
            fprintf(stderr, "[PHASE%d] ERROR: Phase process terminated abnormally\n", phase_num);
            return 1;
        }
    }
}

/* Phase 16: Constraint Validation */
int qallow_phase16_runner(int argc, char** argv) {
    qallow_log_info("PHASE16", "Constraint validation");
    printf("[PHASE16] Constraint validation execution\n");
    return execute_phase_binary(16, argc, argv);
}

/* Phase 17: State Persistence & Checkpointing */
int qallow_phase17_runner(int argc, char** argv) {
    qallow_log_info("PHASE17", "State persistence and checkpointing");
    printf("[PHASE17] State persistence and checkpointing execution\n");
    return execute_phase_binary(17, argc, argv);
}

/* Phase 18: Distributed Execution Coordinator */
int qallow_phase18_runner(int argc, char** argv) {
    qallow_log_info("PHASE18", "Distributed execution coordinator");
    printf("[PHASE18] Distributed execution coordinator execution\n");
    return execute_phase_binary(18, argc, argv);
}

/* Phase 19: Recursive Self-Audit */
int qallow_phase19_runner(int argc, char** argv) {
    qallow_log_info("PHASE19", "Recursive self-audit");
    printf("[PHASE19] Recursive self-audit execution\n");
    return execute_phase_binary(19, argc, argv);
}

/* Phase 20: Quantum Loreweave & Result Synthesis */
int qallow_phase20_runner(int argc, char** argv) {
    qallow_log_info("PHASE20", "Quantum loreweave and result synthesis");
    printf("[PHASE20] Quantum loreweave and result synthesis execution\n");
    return execute_phase_binary(20, argc, argv);
}
