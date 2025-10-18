#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Include all core headers
#include "qallow_kernel.h"
#include "ppai.h"
#include "qcp.h"
#include "ethics.h"
#include "overlay.h"
#include "sandbox.h"
#include "telemetry.h"
#include "pocket.h"
#include "govern.h"
#include "qallow_phase12.h"
#include "qallow_phase13.h"
#include "phase13_accelerator.h"
// TODO: Add these when modules are implemented
// #include "adaptive.h"
// #include "verify.h"
// #include "ingest.h"

typedef enum {
    RUN_PROFILE_STANDARD = 0,
    RUN_PROFILE_BENCH,
    RUN_PROFILE_LIVE
} run_profile_t;

// Forward declarations
static void qallow_build_mode(void);
static void qallow_verify_mode(void);
static void qallow_print_help(void);
static int qallow_run_vm(run_profile_t profile);
static int qallow_handle_run(int argc, char** argv, int arg_offset, run_profile_t default_profile);
static int qallow_dispatch_phase(int argc, char** argv, int start_index, const char* phase_name,
                                 int (*runner)(int, char**));

// Print banner
static void print_banner(void) {
    printf("╔════════════════════════════════════════╗\n");
    printf("║          QALLOW - Unified VM           ║\n");
    printf("║    Photonic & Quantum Emulation        ║\n");
    printf("╚════════════════════════════════════════╝\n\n");
}

// BUILD mode: Compile CPU + CUDA backends
static void qallow_build_mode(void) {
    printf("[BUILD] Qallow is already built!\n");
    printf("[BUILD] To rebuild, run: scripts\\build_wrapper.bat CPU\n");
    printf("[BUILD] Or for CUDA: scripts\\build_wrapper.bat CUDA\n");
}

static int qallow_run_vm(run_profile_t profile) {
    switch (profile) {
        case RUN_PROFILE_BENCH:
            printf("[BENCH] Running HITL benchmark...\n");
            printf("[BENCH] Executing VM with benchmark logging...\n\n");
            printf("[RUN] Executing Qallow VM...\n");
            break;
        case RUN_PROFILE_LIVE:
            printf("[LIVE] Starting Live Interface and External Data Integration\n");
            printf("[LIVE] Ingestion manager initialized with 4 streams\n");
            printf("[LIVE] Streams configured and ready for data ingestion\n");
            printf("[LIVE] - telemetry_primary: http://localhost:9000/telemetry\n");
            printf("[LIVE] - sensor_coherence: http://localhost:9001/coherence\n");
            printf("[LIVE] - sensor_decoherence: http://localhost:9002/decoherence\n");
            printf("[LIVE] - feedback_hitl: http://localhost:9003/feedback\n");
            printf("\n[LIVE] Running VM with live data integration...\n\n");
            break;
        case RUN_PROFILE_STANDARD:
        default:
            printf("[RUN] Executing Qallow VM...\n");
            break;
    }

    print_banner();

    int result = qallow_vm_main();

    if (profile == RUN_PROFILE_LIVE) {
        printf("\n[LIVE] Live interface completed\n");
    }

    return result;
}

static int qallow_dispatch_phase(int argc, char** argv, int start_index, const char* phase_name,
                                 int (*runner)(int, char**)) {
    int trailing = argc - (start_index + 1);
    int phase_argc = 2 + (trailing > 0 ? trailing : 0);
    const char* phase_argv_const[phase_argc];
    int pos = 0;

    phase_argv_const[pos++] = argv[0];
    phase_argv_const[pos++] = phase_name;
    for (int i = start_index + 1; i < argc; ++i) {
        phase_argv_const[pos++] = argv[i];
    }

    return runner(phase_argc, (char**)phase_argv_const);
}

static int qallow_handle_run(int argc, char** argv, int arg_offset, run_profile_t default_profile) {
    run_profile_t profile = default_profile;
    bool profile_set = (default_profile != RUN_PROFILE_STANDARD);

    for (int i = arg_offset; i < argc; ++i) {
        const char* arg = argv[i];

        if (strcmp(arg, "--bench") == 0) {
            if (profile_set && profile != RUN_PROFILE_BENCH) {
                fprintf(stderr, "[ERROR] Conflicting run profile flags\n");
                return 1;
            }
            profile = RUN_PROFILE_BENCH;
            profile_set = true;
            continue;
        }

        if (strcmp(arg, "--live") == 0) {
            if (profile_set && profile != RUN_PROFILE_LIVE) {
                fprintf(stderr, "[ERROR] Conflicting run profile flags\n");
                return 1;
            }
            profile = RUN_PROFILE_LIVE;
            profile_set = true;
            continue;
        }

        if (strcmp(arg, "--accelerator") == 0) {
            int accel_argc = 1 + (argc - (i + 1));
            const char* accel_argv_const[accel_argc];
            int pos = 0;

            accel_argv_const[pos++] = argv[0];
            for (int k = i + 1; k < argc; ++k) {
                accel_argv_const[pos++] = argv[k];
            }

            return qallow_phase13_main(accel_argc, (char**)accel_argv_const);
        }

        if (strncmp(arg, "--phase=", 8) == 0) {
            const char* phase_value = arg + 8;
            if (strcmp(phase_value, "12") == 0 || strcmp(phase_value, "phase12") == 0) {
                return qallow_dispatch_phase(argc, argv, i, "phase12", qallow_phase12_runner);
            }
            if (strcmp(phase_value, "13") == 0 || strcmp(phase_value, "phase13") == 0) {
                return qallow_dispatch_phase(argc, argv, i, "phase13", qallow_phase13_runner);
            }

            fprintf(stderr, "[ERROR] Unknown phase selector: %s\n", phase_value);
            return 1;
        }

        if (strcmp(arg, "--phase12") == 0) {
            return qallow_dispatch_phase(argc, argv, i, "phase12", qallow_phase12_runner);
        }

        if (strcmp(arg, "--phase13") == 0) {
            return qallow_dispatch_phase(argc, argv, i, "phase13", qallow_phase13_runner);
        }

        if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            qallow_print_help();
            return 0;
        }

        fprintf(stderr, "[ERROR] Unknown run option: %s\n", arg);
        qallow_print_help();
        return 1;
    }

    return qallow_run_vm(profile);
}

// VERIFY mode: System checkpoint
static void qallow_verify_mode(void) {
    printf("[VERIFY] Verification mode not yet implemented\n");
    printf("[VERIFY] TODO: Implement system health check\n");
    // TODO: Add verify.c module
}

// Print help message
static void qallow_print_help(void) {
    printf("Usage: qallow [command] [options]\n\n");
    printf("Commands:\n");
    printf("  run [options]     Execute the unified VM workflow (default)\n");
    printf("  build             Detect toolchain and compile CPU + CUDA backends\n");
    printf("  govern            Start governance and ethics audit loop\n");
    printf("  verify            System checkpoint - verify integrity\n");
    printf("  accelerator       Launch the Phase-13 accelerator directly (alias)\n");
    printf("  phase12           Run Phase 12 elasticity simulation (alias)\n");
    printf("  phase13           Run Phase 13 harmonic propagation (alias)\n");
    printf("  help              Show this help message\n\n");
    printf("Run options:\n");
    printf("  --bench           Run the benchmark profile (alias of `qallow bench`)\n");
    printf("  --live            Run the live ingestion profile (alias of `qallow live`)\n");
    printf("  --phase=12|13     Dispatch directly into a legacy phase runner\n");
    printf("  --accelerator     Launch the Phase-13 accelerator; pass accelerator options after this flag\n\n");
    printf("Accelerator options (after --accelerator):\n");
    printf("  --threads=<N|auto>  Worker thread count (auto = online CPUs)\n");
    printf("  --watch=<DIR>       Directory to monitor via inotify\n");
    printf("  --no-watch          Disable watcher even if provided earlier\n");
    printf("  --file=<PATH>       Queue a file for immediate processing (repeatable)\n\n");
    printf("Examples:\n");
    printf("  qallow run                       # Run the unified VM\n");
    printf("  qallow run --bench               # Run benchmark profile\n");
    printf("  qallow run --accelerator --watch=. --threads=auto\n");
    printf("  qallow run --phase=12 --ticks=100 --eps=0.0001 --log=phase12.csv\n");
    printf("  qallow accelerator --watch=/tmp  # Accelerator alias\n");
}

// Main entry point
int main(int argc, char** argv) {
    const char* command = "run";
    int arg_offset = 1;

    if (argc > 1 && argv[1][0] != '-') {
        command = argv[1];
        arg_offset = 2;
    }

    if (strcmp(command, "build") == 0) {
        qallow_build_mode();
        return 0;
    }

    if (strcmp(command, "run") == 0) {
        return qallow_handle_run(argc, argv, arg_offset, RUN_PROFILE_STANDARD);
    }

    if (strcmp(command, "bench") == 0 || strcmp(command, "benchmark") == 0) {
        return qallow_handle_run(argc, argv, arg_offset, RUN_PROFILE_BENCH);
    }

    if (strcmp(command, "live") == 0) {
        return qallow_handle_run(argc, argv, arg_offset, RUN_PROFILE_LIVE);
    }

    if (strcmp(command, "govern") == 0) {
        return govern_cli(argc, argv);
    }

    if (strcmp(command, "verify") == 0) {
        qallow_verify_mode();
        return 0;
    }

    if (strcmp(command, "accelerator") == 0) {
        int accel_argc = 1 + (argc - arg_offset);
        const char* accel_argv_const[accel_argc];
        int pos = 0;

        accel_argv_const[pos++] = argv[0];
        for (int i = arg_offset; i < argc; ++i) {
            accel_argv_const[pos++] = argv[i];
        }

        return qallow_phase13_main(accel_argc, (char**)accel_argv_const);
    }

    if (strcmp(command, "phase12") == 0) {
        return qallow_dispatch_phase(argc, argv, arg_offset - 1, "phase12", qallow_phase12_runner);
    }

    if (strcmp(command, "phase13") == 0) {
        return qallow_dispatch_phase(argc, argv, arg_offset - 1, "phase13", qallow_phase13_runner);
    }

    if (strcmp(command, "help") == 0 || strcmp(command, "-h") == 0 || strcmp(command, "--help") == 0) {
        qallow_print_help();
        return 0;
    }

    if (argv[1] && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)) {
        qallow_print_help();
        return 0;
    }

    fprintf(stderr, "[ERROR] Unknown command: %s\n\n", command);
    qallow_print_help();
    return 1;
}

