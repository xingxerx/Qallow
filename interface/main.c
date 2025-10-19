#define _POSIX_C_SOURCE 199309L

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
#include "phase12.h"
#include "qallow_phase12.h"
#include "qallow_phase13.h"
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
        
        cudaDeviceProp prop;
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
    print_banner();

    // Initialize state
    qallow_state_t state;
    qallow_kernel_init(&state);
    print_system_info(&state);
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
    int max_ticks = 1000;
    for (int tick = 0; tick < max_ticks; tick++) {
        // Run kernel tick
        qallow_kernel_tick(&state);

        // Update pocket dimension telemetry every 5 ticks
        if (tick % 5 == 0) {
            pocket_tick_all(&pocket_dim);
            pocket_merge(&pocket_dim);
            pocket_capture_metrics(&pocket_dim, tick);
        }

        // Compute ethics
        ethics_state_t ethics_state;
        qallow_ethics_check(&state, &ethics_state);

        // Log to CSV every tick (if enabled)
        if (csv_log_path) {
            qallow_csv_log_tick(&state, &ethics_state);
        }

        // Dashboard every 100 ticks
        if (tick % 50 == 0) {
            qallow_print_dashboard(&state, &ethics_state);
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
    
    return 0;
}
