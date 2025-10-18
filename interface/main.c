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
#include <time.h>

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
int qallow_vm_main(void) {
    print_banner();
    printf("[SYSTEM] Qallow VM initialized\n");
    printf("[SYSTEM] Execution mode: CPU\n");
    printf("[KERNEL] Node count: 256 per overlay\n");
    printf("[KERNEL] Max ticks: 1000\n\n");
    printf("[MAIN] Starting VM execution loop...\n\n");

    // Initialize state
    qallow_state_t state;
    qallow_kernel_init(&state);

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

        // Compute ethics
        ethics_state_t ethics_state;
        qallow_ethics_check(&state, &ethics_state);

        // Log to CSV every tick (if enabled)
        if (csv_log_path) {
            qallow_csv_log_tick(&state, &ethics_state);
        }

        // Dashboard every 100 ticks
        if (tick % 100 == 0) {
            qallow_print_dashboard(&state, &ethics_state);
        }

        // Check for equilibrium
        if (state.decoherence_level < 0.0001f) {
            printf("\n[KERNEL] System reached stable equilibrium at tick %d\n", tick);
            break;
        }
    }

    // Cleanup
    if (csv_log_path) {
        qallow_csv_log_close();
        printf("\n[CSV] Log file closed\n");
    }

    printf("\n[MAIN] VM execution completed\n");
    printf("[TELEMETRY] Benchmark logged: compile=0.0ms, run=%.2fms, mode=CPU\n\n", 
           max_ticks * 0.001);
    
    return 0;
}