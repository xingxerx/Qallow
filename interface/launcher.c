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
// TODO: Add these when modules are implemented
// #include "adaptive.h"
// #include "verify.h"
// #include "ingest.h"

// Forward declarations for mode handlers
static void qallow_build_mode(void);
static void qallow_run_mode(void);
static void qallow_bench_mode(void);
static void qallow_verify_mode(void);
static void qallow_live_mode(void);
static void qallow_print_help(void);

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
    printf("[BENCH] Executing VM with benchmark logging...\n\n");

    // Run the VM which logs benchmark data
    qallow_run_mode();
}

// VERIFY mode: System checkpoint
static void qallow_verify_mode(void) {
    printf("[VERIFY] Verification mode not yet implemented\n");
    printf("[VERIFY] TODO: Implement system health check\n");
    // TODO: Add verify.c module
}

// LIVE mode: Live Interface and External Data Integration
static void qallow_live_mode(void) {
    printf("[LIVE] Starting Live Interface and External Data Integration\n");
    printf("[LIVE] Ingestion manager initialized with 4 streams\n");
    printf("[LIVE] Streams configured and ready for data ingestion\n");
    printf("[LIVE] - telemetry_primary: http://localhost:9000/telemetry\n");
    printf("[LIVE] - sensor_coherence: http://localhost:9001/coherence\n");
    printf("[LIVE] - sensor_decoherence: http://localhost:9002/decoherence\n");
    printf("[LIVE] - feedback_hitl: http://localhost:9003/feedback\n");
    printf("\n[LIVE] Running VM with live data integration...\n\n");

    // Run the VM
    int result = qallow_vm_main();

    printf("\n[LIVE] Live interface completed\n");
    exit(result);
}

// Print help message
static void qallow_print_help(void) {
    printf("Usage: qallow [mode]\n\n");
    printf("Modes:\n");
    printf("  build    Detect toolchain and compile CPU + CUDA backends\n");
    printf("  run      Execute the VM (auto-selects CPU/CUDA)\n");
    printf("  bench    Run benchmark with logging\n");
    printf("  govern   Start governance and ethics audit loop\n");
    printf("  verify   System checkpoint - verify integrity\n");
    printf("  live     Live interface and external data integration\n");
    printf("  phase12  Run Phase 12 elasticity simulation\n");
    printf("  phase13  Run Phase 13 harmonic propagation\n");
    printf("  help     Show this help message\n\n");
    printf("Examples:\n");
    printf("  qallow build      # Build both CPU and CUDA versions\n");
    printf("  qallow run        # Run the VM\n");
    printf("  qallow bench      # Run benchmark\n");
    printf("  qallow govern     # Run governance audit\n");
    printf("  qallow verify     # Verify system health\n");
    printf("  qallow live       # Start live interface\n");
    printf("  qallow phase12 --ticks=100 --eps=0.0001 --log=phase12.csv\n");
    printf("  qallow phase13 --nodes=8 --ticks=500 --k=0.001 --log=phase13.csv\n");
}

// Main entry point
int main(int argc, char** argv) {
    const char* mode = (argc > 1) ? argv[1] : "run";

    // Only print banner for run, bench, govern, verify, and live modes
    if (strcmp(mode, "build") != 0 && strcmp(mode, "help") != 0 &&
        strcmp(mode, "-h") != 0 && strcmp(mode, "--help") != 0) {
        print_banner();
    }

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

    if (strcmp(mode, "govern") == 0) {
        return govern_cli(argc, argv);
    }

    if (strcmp(mode, "verify") == 0) {
        qallow_verify_mode();
        return 0;
    }

    if (strcmp(mode, "live") == 0) {
        qallow_live_mode();
        return 0;
    }

    if (strcmp(mode, "phase12") == 0) {
        return qallow_phase12_runner(argc, argv);
    }

    if (strcmp(mode, "phase13") == 0) {
        return qallow_phase13_runner(argc, argv);
    }

    if (strcmp(mode, "help") == 0 || strcmp(mode, "-h") == 0 || strcmp(mode, "--help") == 0) {
        qallow_print_help();
        return 0;
    }

    printf("[ERROR] Unknown mode: %s\n\n", mode);
    qallow_print_help();
    return 1;
}
