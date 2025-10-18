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
#include "adaptive.h"
#include "pocket.h"
#include "govern.h"

// Forward declarations for mode handlers
static void qallow_build_mode(void);
static void qallow_run_mode(void);
static void qallow_bench_mode(void);
static void qallow_visual_mode(void);
static void qallow_govern_mode(void);
static void qallow_print_help(void);

// Print banner
static void print_banner(void) {
    printf("╔════════════════════════════════════════╗\n");
    printf("║     QALLOW - Unified Command System    ║\n");
    printf("║  Autonomous Governance & Optimization  ║\n");
    printf("╚════════════════════════════════════════╝\n\n");
}

// BUILD mode: Compile CPU + CUDA backends
static void qallow_build_mode(void) {
    printf("[BUILD] Detecting toolchain...\n");
    
    // Check for CUDA
    int cuda_available = system("nvcc --version >nul 2>&1") == 0;
    
    if (cuda_available) {
        printf("[BUILD] CUDA detected - building CUDA-accelerated version...\n");
        printf("[BUILD] Command: scripts\\build_wrapper.bat CUDA\n");
        int result = system("scripts\\build_wrapper.bat CUDA");
        if (result == 0) {
            printf("[BUILD] ✓ CUDA build completed successfully\n");
        } else {
            printf("[BUILD] ✗ CUDA build failed, falling back to CPU\n");
        }
    } else {
        printf("[BUILD] CUDA not detected - building CPU-only version...\n");
        printf("[BUILD] Command: scripts\\build_wrapper.bat CPU\n");
        int result = system("scripts\\build_wrapper.bat CPU");
        if (result == 0) {
            printf("[BUILD] ✓ CPU build completed successfully\n");
        } else {
            printf("[BUILD] ✗ CPU build failed\n");
            exit(1);
        }
    }
}

// RUN mode: Execute current binary
static void qallow_run_mode(void) {
    printf("[RUN] Executing Qallow VM...\n");

    // Call the VM main function directly
    int result = qallow_vm_main();
    exit(result);
}

// BENCH mode: Run HITL benchmark
static void qallow_bench_mode(void) {
    printf("[BENCH] Running HITL benchmark...\n");
    printf("[BENCH] Executing benchmark script...\n");
    
    int result = system("powershell -ExecutionPolicy Bypass -File scripts\\benchmark.ps1");
    if (result == 0) {
        printf("[BENCH] ✓ Benchmark completed\n");
    } else {
        printf("[BENCH] ✗ Benchmark failed\n");
        exit(1);
    }
}

// VISUAL mode: Open live dashboard
static void qallow_visual_mode(void) {
    printf("[VISUAL] Opening live dashboard...\n");
    printf("[VISUAL] Dashboard URL: http://localhost:8080\n");
    printf("[VISUAL] Note: Dashboard server not yet implemented\n");
    printf("[VISUAL] For now, view telemetry data in qallow_stream.csv\n");
}

// GOVERN mode: Run autonomous governance loop
static void qallow_govern_mode(void) {
    printf("[GOVERN] Starting autonomous governance loop...\n\n");
    
    // Initialize all subsystems
    govern_state_t gov;
    govern_init(&gov);
    
    qallow_state_t state;
    qallow_kernel_init(&state);
    
    ethics_monitor_t ethics;
    ethics_init(&ethics);
    
    sandbox_manager_t sandbox;
    sandbox_init(&sandbox);
    
    adaptive_state_t adaptive;
    adaptive_load(&adaptive);
    
    // Run the governance loop
    govern_run_audit_loop(&gov, &state, &ethics, &sandbox, &adaptive);
    
    // Cleanup
    sandbox_cleanup(&sandbox);
    
    printf("\n[GOVERN] Autonomous governance completed\n");
}

// Print help message
static void qallow_print_help(void) {
    printf("Usage: qallow [mode]\n\n");
    printf("Modes:\n");
    printf("  build    Detect toolchain and compile CPU + CUDA backends\n");
    printf("  run      Execute current binary (auto-selects CPU/CUDA)\n");
    printf("  bench    Run HITL benchmark with logging\n");
    printf("  visual   Open live dashboard\n");
    printf("  govern   Start autonomous governance and ethics audit loop\n");
    printf("  help     Show this help message\n\n");
    printf("Examples:\n");
    printf("  qallow build      # Build both CPU and CUDA versions\n");
    printf("  qallow run        # Run the VM\n");
    printf("  qallow bench      # Run benchmark\n");
    printf("  qallow govern     # Run governance audit\n");
}

// Main entry point
int main(int argc, char** argv) {
    print_banner();
    
    const char* mode = (argc > 1) ? argv[1] : "run";
    
    if (strcmp(mode, "build") == 0) {
        qallow_build_mode();
        return 0;
    }
    
    if (strcmp(mode, "run") == 0) {
        qallow_run_mode();
        return 0;
    }
    
    if (strcmp(mode, "bench") == 0 || strcmp(mode, "benchmark") == 0) {
        qallow_bench_mode();
        return 0;
    }
    
    if (strcmp(mode, "visual") == 0 || strcmp(mode, "dashboard") == 0) {
        qallow_visual_mode();
        return 0;
    }
    
    if (strcmp(mode, "govern") == 0) {
        qallow_govern_mode();
        return 0;
    }
    
    if (strcmp(mode, "help") == 0 || strcmp(mode, "-h") == 0 || strcmp(mode, "--help") == 0) {
        qallow_print_help();
        return 0;
    }
    
    printf("[ERROR] Unknown mode: %s\n\n", mode);
    qallow_print_help();
    return 1;
}

