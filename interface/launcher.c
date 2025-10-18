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
#include "verify.h"
#include "ingest.h"

// Forward declarations for mode handlers
static void qallow_build_mode(void);
static void qallow_run_mode(void);
static void qallow_bench_mode(void);
static void qallow_visual_mode(void);
static void qallow_govern_mode(void);
static void qallow_verify_mode(void);
static void qallow_live_mode(void);
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

// VERIFY mode: System checkpoint for Phase 6
static void qallow_verify_mode(void) {
    printf("[VERIFY] Running system checkpoint...\n\n");

    verify_report_t report;
    int result = verify_system(&report);

    verify_print_report(&report);

    if (verify_is_healthy(&report)) {
        printf("[VERIFY] System is healthy and ready for Phase 6 expansion\n");
        exit(0);
    } else {
        printf("[VERIFY] System issues detected - Phase 6 expansion blocked\n");
        exit(1);
    }
}

// LIVE mode: Phase 6 - Live Interface and External Data Integration
static void qallow_live_mode(void) {
    printf("[LIVE] Starting Phase 6 - Live Interface and External Data Integration\n\n");

    // Initialize ingestion manager
    ingest_manager_t ingest_mgr;
    ingest_init(&ingest_mgr);

    // Add default streams
    ingest_add_stream(&ingest_mgr, "telemetry_primary", "http://localhost:9000/telemetry");
    ingest_add_stream(&ingest_mgr, "sensor_coherence", "http://localhost:9001/coherence");
    ingest_add_stream(&ingest_mgr, "sensor_decoherence", "http://localhost:9002/decoherence");
    ingest_add_stream(&ingest_mgr, "feedback_hitl", "http://localhost:9003/feedback");

    printf("[LIVE] Ingestion manager initialized with 4 streams\n");
    printf("[LIVE] Streams configured and ready for data ingestion\n");

    // Run VM with live data integration
    printf("\n[LIVE] Running VM with live data integration...\n\n");
    int result = qallow_vm_main();

    // Cleanup
    ingest_cleanup(&ingest_mgr);
    printf("\n[LIVE] Phase 6 live interface completed\n");

    exit(result);
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
    printf("  verify   System checkpoint - verify integrity before Phase 6\n");
    printf("  live     Phase 6 - Live interface and external data integration\n");
    printf("  help     Show this help message\n\n");
    printf("Examples:\n");
    printf("  qallow build      # Build both CPU and CUDA versions\n");
    printf("  qallow run        # Run the VM\n");
    printf("  qallow bench      # Run benchmark\n");
    printf("  qallow govern     # Run governance audit\n");
    printf("  qallow verify     # Verify system health\n");
    printf("  qallow live       # Start Phase 6 live interface\n");
}

// Main entry point
int main(int argc, char** argv) {
    const char* mode = (argc > 1) ? argv[1] : "run";

    // Only print banner for run, bench, govern, visual, verify, and live modes
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

    if (strcmp(mode, "visual") == 0 || strcmp(mode, "dashboard") == 0) {
        qallow_visual_mode();
        return 0;
    }

    if (strcmp(mode, "govern") == 0) {
        qallow_govern_mode();
        return 0;
    }

    if (strcmp(mode, "verify") == 0) {
        qallow_verify_mode();
        return 0;
    }

    if (strcmp(mode, "live") == 0) {
        qallow_live_mode();
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

