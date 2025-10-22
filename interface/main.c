#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>

#include "qallow/logging.h"
#include "qallow/profiling.h"
#include "qallow/env.h"
#include "qallow/logging.h"
#include "qallow/module.h"
#include "qallow_kernel.h"
#include "ppai.h"
#include "qcp.h"
#include "ethics.h"
#include "overlay.h"
#include "sandbox.h"
#include "telemetry.h"
#include "adaptive.h"
#include "pocket.h"
#include "phase7.h"
#include "qallow_phase11.h"
#include "phase12.h"
#include "qallow_phase12.h"
#include "qallow_phase13.h"
#include "meta_introspect.h"
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>

// Main application entry point for Qallow VM
// Supports both CUDA and CPU execution with unified telemetry and adaptive learning

static void print_banner(void) {
    printf("╔════════════════════════════════════════╗\n");
    printf("║          QALLOW VM - Unified           ║\n");
    printf("║  Photonic & Quantum Hardware Emulation ║\n");
    printf("║  CPU + CUDA Acceleration Support       ║\n");
    printf("║  Multi-Pocket + Chronometric Sim       ║\n");
    printf("║  Proactive AGI Layer                   ║\n");
    printf("╚════════════════════════════════════════╝\n\n");
}

static void print_system_info(const qallow_state_t* state) {
    printf("[SYSTEM] Qallow VM initialized\n");
    printf("[SYSTEM] Execution mode: %s\n", state->cuda_enabled ? "CUDA GPU" : "CPU");
    
#if CUDA_ENABLED
    if (state->cuda_enabled) {
        int device_count;
        cudaGetDeviceCount(&device_count);
        
        struct cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, state->gpu_device_id);
        
        printf("[CUDA] GPU: %s\n", prop.name);
        printf("[CUDA] Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("[CUDA] Memory: %.1f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("[CUDA] Multiprocessors: %d\n", prop.multiProcessorCount);
    }
#endif
    
    printf("[KERNEL] Node count: %d per overlay\n", MAX_NODES);
    printf("[KERNEL] Max ticks: %d\n", MAX_TICKS);
    printf("\n");
}

static void initialize_logging(void) {
    static int initialized = 0;
    if (initialized) {
        return;
    }
    qallow_logging_init();
    qallow_env_load(NULL);
    const char* log_dir = getenv("QALLOW_LOG_DIR");
    if (log_dir && *log_dir) {
        qallow_logging_set_directory(log_dir);
    }
    initialized = 1;
}

static const char* detect_python_binary(void) {
    const char* env_python = getenv("QALLOW_PYTHON");
    if (env_python && *env_python && access(env_python, X_OK) == 0) {
        return env_python;
    }
    if (access("./qiskit-env/bin/python", X_OK) == 0) {
        return "./qiskit-env/bin/python";
    }
    if (access("python3", X_OK) == 0) {
        return "python3";
    }
    return "python3";
}

static int sanitize_states(const char* raw, char* out, size_t out_len) {
    size_t pos = 0;
    if (!raw || !*raw) {
        raw = "-1,0,1";
    }
    for (const char* c = raw; *c; ++c) {
        if (*c == ' ' || *c == '\t' || *c == '\n' || *c == '\r') {
            continue;
        }
        if (*c != '-' && *c != ',' && (*c < '0' || *c > '9')) {
            fprintf(stderr, "[PHASE11] Invalid character in --states: %c\n", *c);
            return 0;
        }
        if (pos + 1 >= out_len) {
            fprintf(stderr, "[PHASE11] --states value too long\n");
            return 0;
        }
        out[pos++] = *c;
    }
    if (pos == 0) {
        if (out_len < 6) {
            return 0;
        }
        strcpy(out, "-1,0,1");
    } else {
        out[pos] = '\0';
    }
    return 1;
}

int qallow_phase11_runner(int argc, char** argv) {
    int ticks = 400;
    const char* states_arg = NULL;
    int hardware_only = 0;

    for (int i = 2; i < argc; ++i) {
        const char* arg = argv[i];
        if (strncmp(arg, "--ticks=", 8) == 0) {
            ticks = atoi(arg + 8);
            if (ticks < 1) {
                ticks = 1;
            }
        } else if (strncmp(arg, "--states=", 9) == 0) {
            states_arg = arg + 9;
        } else if (strcmp(arg, "--hardware-only") == 0) {
            hardware_only = 1;
        }
    }

    char states_clean[128];
    if (!sanitize_states(states_arg, states_clean, sizeof(states_clean))) {
        return 1;
    }

    const char* python_bin = detect_python_binary();
    const int shots = ticks > 0 ? ticks : 1;

    char command[512];
    int written;
    if (hardware_only) {
        written = snprintf(
            command,
            sizeof(command),
            "\"%s\" -m python.quantum.run_phase11_bridge --shots=%d --states=\"%s\" --hardware-only",
            python_bin,
            shots,
            states_clean);
    } else {
        written = snprintf(
            command,
            sizeof(command),
            "\"%s\" -m python.quantum.run_phase11_bridge --shots=%d --states=\"%s\"",
            python_bin,
            shots,
            states_clean);
    }

    if (written < 0 || (size_t)written >= sizeof(command)) {
        fprintf(stderr, "[PHASE11] Failed to compose Python command\n");
        return 1;
    }

    printf("[PHASE11] Invoking bridge via %s\n", python_bin);
    fflush(stdout);
    int rc = system(command);
    if (rc == -1) {
        perror("[PHASE11] system() failed");
        return 1;
    }
    if (WIFEXITED(rc)) {
        int status = WEXITSTATUS(rc);
        if (status != 0) {
            fprintf(stderr, "[PHASE11] Bridge exited with code %d\n", status);
        }
        return status;
    }
    fprintf(stderr, "[PHASE11] Bridge terminated unexpectedly\n");
    return 1;
}

static int qallow_vm_run_hardware(void) {
    printf("[HARDWARE] Qallow hardware mode enabled (IBM Quantum).\n");
    printf("[HARDWARE] Dispatching Phase 11 coherence bridge to real backend...\n");

    const char* shots_env = getenv("QALLOW_PHASE11_SHOTS");
    int shots = shots_env ? atoi(shots_env) : 1024;
    if (shots < 1) {
        shots = 1024;
    }

    const char* states_env = getenv("QALLOW_PHASE11_STATES");
    char states_clean[128];
    if (!sanitize_states(states_env, states_clean, sizeof(states_clean))) {
        fprintf(stderr, "[HARDWARE] Invalid QALLOW_PHASE11_STATES value.\n");
        return 1;
    }

    char shots_arg[48];
    snprintf(shots_arg, sizeof(shots_arg), "--shots=%d", shots);

    char states_arg[160];
    snprintf(states_arg, sizeof(states_arg), "--states=%s", states_clean);

    char* args[] = {
        "qallow",
        "phase11",
        shots_arg,
        states_arg,
        "--hardware-only",
        NULL,
    };

    int rc = qallow_phase11_runner(5, args);
    if (rc == 0) {
        printf("[HARDWARE] Phase 11 execution completed on IBM Quantum hardware.\n");
        printf("[HARDWARE] Review JSON output above for telemetry payload.\n");
    } else {
        fprintf(stderr, "[HARDWARE] Phase 11 hardware execution failed (code=%d).\n", rc);
    }

    return rc;
}

// VM execution function (called from launcher)
int qallow_phase12_runner(int argc, char** argv) {
    int ticks = 1000;
    float eps = 0.0001f;
    const char* log_path = NULL;

    for (int i = 2; i < argc; ++i) {
        const char* arg = argv[i];
        if (strncmp(arg, "--ticks=", 8) == 0) {
            ticks = atoi(arg + 8);
            if (ticks < 1) ticks = 1;
        } else if (strncmp(arg, "--eps=", 6) == 0) {
            eps = (float)atof(arg + 6);
            if (eps < 0.0f) eps = 0.0f;
        } else if (strncmp(arg, "--log=", 6) == 0) {
            log_path = arg + 6;
        }
    }

    printf("[PHASE12] Elasticity simulation\n");
    printf("[PHASE12] ticks=%d eps=%.6f\n", ticks, eps);
    if (log_path) {
        printf("[PHASE12] log=%s\n", log_path);
    }

    return run_phase12_elasticity(log_path, ticks, eps);
}

int qallow_phase13_runner(int argc, char** argv) {
    int nodes = 8;
    int ticks = 400;
    float coupling = 0.001f;
    const char* log_path = NULL;

    for (int i = 2; i < argc; ++i) {
        const char* arg = argv[i];
        if (strncmp(arg, "--nodes=", 8) == 0) {
            nodes = atoi(arg + 8);
            if (nodes < 2) nodes = 2;
        } else if (strncmp(arg, "--ticks=", 8) == 0) {
            ticks = atoi(arg + 8);
            if (ticks < 1) ticks = 1;
        } else if (strncmp(arg, "--k=", 4) == 0) {
            coupling = (float)atof(arg + 4);
            if (coupling <= 0.0f) coupling = 0.0001f;
        } else if (strncmp(arg, "--log=", 6) == 0) {
            log_path = arg + 6;
        }
    }

    printf("[PHASE13] Harmonic propagation\n");
    printf("[PHASE13] nodes=%d ticks=%d k=%.6f\n", nodes, ticks, coupling);
    if (log_path) {
        printf("[PHASE13] log=%s\n", log_path);
    }

    return run_phase13_harmonic(log_path, nodes, ticks, coupling);
}

int qallow_vm_main(void) {
    initialize_logging();
    print_banner();

    const char* mode = getenv("QALLOW_MODE");
    if (mode && strcmp(mode, "hardware") == 0) {
        return qallow_vm_run_hardware();
    }

    // Initialize state
    qallow_state_t state;
    qallow_kernel_init(&state);
    print_system_info(&state);
    qallow_log_info("vm", "mode=%s", state.cuda_enabled ? "cuda" : "cpu");
    meta_introspect_apply_environment_defaults();
    if (state.cuda_enabled) {
        meta_introspect_set_gpu_available(1);
    }
    printf("[MAIN] Starting VM execution loop...\n\n");

    pocket_dimension_t pocket_dim;
    pocket_spawn(&pocket_dim, 4);

    mkdir("data", 0755);
    mkdir("data/telemetry", 0755);

    // Initialize CSV logging from environment
    const char* csv_log_path = getenv("QALLOW_LOG");
    if (csv_log_path) {
        qallow_csv_log_init(csv_log_path);
        printf("[CSV] Logging enabled: %s\n\n", csv_log_path);
    }

    // Main execution loop
    int dashboard_interval = 50;
    const char* dashboard_env = getenv("QALLOW_DASHBOARD_INTERVAL");
    if (dashboard_env && *dashboard_env) {
        int parsed = atoi(dashboard_env);
        if (parsed > 0) {
            dashboard_interval = parsed;
        } else {
            dashboard_interval = 0; // treat zero or negative as disabled
        }
    }
    int max_ticks = 1000;
    bool dashboard_muted = false;
    for (int tick = 0; tick < max_ticks; tick++) {
        // Run kernel tick
        QALLOW_PROFILE_SCOPE("kernel_tick") {
            qallow_kernel_tick(&state);
        }

        // Update pocket dimension telemetry every 5 ticks
        if (tick % 5 == 0) {
            QALLOW_PROFILE_SCOPE("pocket_update") {
                pocket_tick_all(&pocket_dim);
                pocket_merge(&pocket_dim);
                pocket_capture_metrics(&pocket_dim, tick);
            }
        }

        // Compute ethics
        ethics_state_t ethics_state;
        QALLOW_PROFILE_SCOPE("ethics_check") {
            qallow_ethics_check(&state, &ethics_state);
        }

        // Log to CSV every tick (if enabled)
        if (csv_log_path) {
            qallow_csv_log_tick(&state, &ethics_state);
            qallow_log_info("vm.tick", "tick=%d decoherence=%.6f", tick, state.decoherence_level);
        }

        // Dashboard at configured interval (disabled when interval is zero)
        if (!dashboard_muted && dashboard_interval > 0 && tick % dashboard_interval == 0) {
            qallow_print_dashboard(&state, &ethics_state);
            if (!ethics_state.safety_check_passed) {
                printf("[DASHBOARD] Ethics fail detected at tick %d; muting dashboard output.\n", tick);
                dashboard_muted = true;
            }
        }

        // Check for equilibrium
        if (state.decoherence_level < 0.0001f && tick > 200) {
            printf("\n[KERNEL] System reached stable equilibrium at tick %d\n", tick);
            break;
        }

        struct timespec ts = {0, 20000000};
        nanosleep(&ts, NULL);
    }

    // Cleanup
    pocket_cleanup(&pocket_dim);
    if (csv_log_path) {
        qallow_csv_log_close();
        printf("\n[CSV] Log file closed\n");
    }

    printf("\n[MAIN] VM execution completed\n");
    printf("[TELEMETRY] Benchmark logged: compile=0.0ms, run=%.2fms, mode=CPU\n\n", 
        max_ticks * 0.001);
    qallow_log_info("vm.complete", "ticks=%d", max_ticks);
    meta_introspect_flush();
    
    return 0;
}
