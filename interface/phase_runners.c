#include "phase_runners.h"
#include "qallow/logging.h"
#include "qallow_phase11.h"
#include "qallow_phase12.h"
#include "qallow_phase13.h"
#include "qallow_phase14.h"
#include "qallow_phase15.h"
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

/* Phase 11: Coherence Bridge */
int qallow_phase11_runner(int argc, char** argv) {
    qallow_log_info("PHASE11", "Coherence bridge runner");
    printf("[PHASE11] Coherence bridge execution\n");
    return 0;
}

/* Phase 12: Elasticity Simulation */
int qallow_phase12_runner(int argc, char** argv) {
    int ticks = 500;
    const char* audit_tag = "phase12";
    
    // Parse command-line args if provided
    for (int i = 2; i < argc; ++i) {
        if (strncmp(argv[i], "--ticks=", 8) == 0) {
            ticks = atoi(argv[i] + 8);
        }
    }
    
    qallow_log_info("BENCHMARK", "Calling real: qallow_phase12_runner");
    return run_phase12_elasticity(audit_tag, NULL, ticks, 0.1f);
}

/* Phase 13: Harmonic Propagation */
int qallow_phase13_runner(int argc, char** argv) {
    int ticks = 500;
    int num_nodes = 256;
    const char* audit_tag = "phase13";
    
    // Parse command-line args if provided
    for (int i = 2; i < argc; ++i) {
        if (strncmp(argv[i], "--ticks=", 8) == 0) {
            ticks = atoi(argv[i] + 8);
        } else if (strncmp(argv[i], "--nodes=", 8) == 0) {
            num_nodes = atoi(argv[i] + 8);
        }
    }
    
    qallow_log_info("BENCHMARK", "Calling real: qallow_phase13_runner");
    return run_phase13_harmonic(audit_tag, NULL, num_nodes, ticks, 0.5f);
}

/* Phase 14: Coherence-Lattice Integration */
int qallow_phase14_runner(int argc, char** argv) {
    qallow_log_info("PHASE14", "Coherence-lattice integration");
    printf("[PHASE14] Coherence-lattice integration execution\n");
    return 0;
}

/* Phase 15: Convergence & Lock-in */
int qallow_phase15_runner(int argc, char** argv) {
    qallow_log_info("PHASE15", "Convergence and lock-in");
    printf("[PHASE15] Convergence and lock-in execution\n");
    return 0;
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
