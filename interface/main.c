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
    printf("║     QALLOW VM - Unified System         ║\n");
    printf("║  Photonic & Quantum Hardware Emulation ║\n");
    printf("║  CPU + CUDA Acceleration Support       ║\n");
    printf("║  Phase IV: Multi-Pocket + Chronometric ║\n");
    printf("║  Phase VII: Proactive AGI Layer        ║\n");
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

    // Start timing
    clock_t start_time = clock();

    // Initialize telemetry system
    telemetry_t telemetry;
    telemetry_init(&telemetry);

    // Initialize adaptive reinforcement system
    adaptive_state_t adaptive;
    adaptive_load(&adaptive);

    // Initialize core state
    qallow_state_t state;
    qallow_kernel_init(&state);

    // Initialize subsystems
    ppai_state_t ppai;
    ppai_init(&ppai);

    qcp_state_t qcp;
    qcp_init(&qcp);

    ethics_monitor_t ethics;
    ethics_init(&ethics);

    sandbox_manager_t sandbox;
    sandbox_init(&sandbox);

    // Initialize pocket dimension simulator
    pocket_dimension_t pocket_dim;
    memset(&pocket_dim, 0, sizeof(pocket_dim));

    // ========================================================================
    // Initialize Phase 7: Proactive AGI Layer
    // ========================================================================
    phase7_state_t phase7;
    if (phase7_init(&phase7, "data") == 0) {
        printf("[PHASE7] Proactive AGI Layer active\n");
    } else {
        printf("[PHASE7] Warning: Failed to initialize, continuing without Phase 7\n");
        memset(&phase7, 0, sizeof(phase7));
    }

    print_system_info(&state);
    
    // Create initial safety snapshot
    sandbox_create_snapshot(&sandbox, &state, "initial_safe_state");
    
    printf("[MAIN] Starting VM execution loop...\n\n");

    // Spawn pocket dimension simulations (optional, every 100 ticks)
    int pocket_active = 0;

    // Main execution loop
    for (int tick = 0; tick < MAX_TICKS; tick++) {
        // Run one VM tick
        qallow_kernel_tick(&state);

        // Process PPAI (Photonic-Probabilistic AI)
#if CUDA_ENABLED
        if (state.cuda_enabled) {
            ppai_cuda_process_photons(&ppai, state.overlays, NUM_OVERLAYS);
        } else {
#endif
            ppai_cpu_process_photons(&ppai, state.overlays, NUM_OVERLAYS);
#if CUDA_ENABLED
        }
#endif

        // Process QCP (Quantum Co-Processor)
#if CUDA_ENABLED
        if (state.cuda_enabled) {
            qcp_cuda_process_qubits(&qcp, state.overlays, NUM_OVERLAYS);
        } else {
#endif
            qcp_cpu_process_qubits(&qcp, state.overlays, NUM_OVERLAYS);
#if CUDA_ENABLED
        }
#endif

        // CUDA-accelerated pocket dimension simulation
#if CUDA_ENABLED
        if (state.cuda_enabled && tick > 0 && tick % 400 == 0) {
            printf("[CUDA] Spawning accelerated pocket dimension simulation...\n");
            pocket_cfg_t pcfg = {.pockets=32, .nodes=256, .steps=200, .jitter=0.0005};
            if(pocket_spawn_and_run(&pcfg)==0){
                // merge back to host arrays (length=nodes)
                double *O = (double*)malloc(pcfg.nodes*sizeof(double));
                double *R = (double*)malloc(pcfg.nodes*sizeof(double));
                double *M = (double*)malloc(pcfg.nodes*sizeof(double));
                pocket_merge_to_host(O,R,M);
                // integrate O/R/M means into your global coherence, telemetry, E=S+C+H, etc.
                // For now, just printing a placeholder message
                printf("[CUDA] Pocket dimension results merged to host.\n");
                free(O); free(R); free(M);
                pocket_release();
            }
        }
#endif

        // Stream telemetry data
        telemetry_stream_tick(&telemetry,
                             state.overlays[0].stability,
                             state.overlays[1].stability,
                             state.overlays[2].stability,
                             state.global_coherence,
                             state.decoherence_level,
                             state.cuda_enabled ? 1 : 0);

        // Ethics monitoring and safety checks
        if (!ethics_evaluate_state(&state, &ethics)) {
            printf("[ETHICS] Safety violation detected at tick %d!\n", tick);

            // Attempt rollback to safe state
            if (sandbox_rollback_to_safe_state(&sandbox, &state)) {
                printf("[SANDBOX] Successfully rolled back to safe state\n");
                break;
            } else {
                printf("[EMERGENCY] Emergency shutdown initiated\n");
                ethics_emergency_shutdown(&state, "Critical safety violation");
                telemetry_close(&telemetry);
                return 1;
            }
        }

        // Print status every 100 ticks or on interesting changes
        if (tick % 100 == 0 || tick < 10) {
            qallow_print_status(&state, tick);
        }

        // Spawn pocket dimension simulations every 200 ticks
        if (tick % 200 == 0 && tick > 0 && !pocket_active) {
            pocket_spawn(&pocket_dim, 4);
            pocket_active = 1;
        }

        // Run pocket simulations
        if (pocket_active) {
            pocket_tick_all(&pocket_dim);
        }

        // Merge pocket results every 50 ticks
        if (pocket_active && tick % 50 == 0 && tick > 0) {
            double pocket_score = pocket_merge(&pocket_dim);
            adaptive_update(&adaptive, 0.0, pocket_score);
        }

        // Create periodic snapshots
        if (tick % 500 == 0 && tick > 0) {
            char snapshot_name[64];
            snprintf(snapshot_name, sizeof(snapshot_name), "auto_snapshot_tick_%d", tick);
            sandbox_create_snapshot(&sandbox, &state, snapshot_name);
        }

        // ====================================================================
        // Phase 7 Tick: Goal Synthesis -> Planning -> Execution -> Reflection
        // ====================================================================
        if (phase7.phase7_active && tick % 100 == 0) {
            phase7_tick(&phase7, &telemetry, &ethics);
            
            // Check hard stops
            if (phase7_check_hard_stops(&phase7, &ethics)) {
                printf("[PHASE7] Hard stop triggered - emergency shutdown\n");
                break;
            }
        }

        // Early termination if system reaches stable equilibrium
        if (state.global_coherence > 0.999f && state.decoherence_level < 0.0001f) {
            printf("[KERNEL] System reached stable equilibrium at tick %d\n", tick);
            break;
        }
    }
    
    printf("\n[MAIN] VM execution completed\n");

    // Calculate total runtime
    clock_t end_time = clock();
    double run_ms = ((double)(end_time - start_time) / CLOCKS_PER_SEC) * 1000.0;

    // Log final benchmark
    telemetry_log_benchmark(&telemetry, 0.0, run_ms,
                           state.decoherence_level, state.global_coherence,
                           state.cuda_enabled ? 1 : 0);

    // Final reports
    printf("\n═══ FINAL STATUS REPORT ═══\n");
    qallow_print_status(&state, state.tick_count);

    printf("\n═══ ETHICS REPORT ═══\n");
    ethics_print_report(&ethics);

    printf("\n═══ SANDBOX REPORT ═══\n");
    sandbox_print_resource_report(&sandbox);
    sandbox_list_snapshots(&sandbox);

    printf("\n═══ ADAPTIVE STATE ═══\n");
    printf("[ADAPTIVE] Target: %.1fms, Last run: %.2fms\n", adaptive.target_ms, adaptive.last_run_ms);
    printf("[ADAPTIVE] Threads: %d, Learning rate: %.4f\n", adaptive.threads, adaptive.learning_rate);
    printf("[ADAPTIVE] Human score: %.2f\n", adaptive.human_score);

    printf("\n═══ POCKET DIMENSION REPORT ═══\n");
    if (pocket_active) {
        printf("[POCKET] Final merged score: %.4f\n", pocket_get_average_score(&pocket_dim));
        pocket_cleanup(&pocket_dim);
    }

    // Phase 7 final report and audit
    if (phase7.phase7_active) {
        printf("\n═══ PHASE 7 PROACTIVE AGI REPORT ═══\n");
        phase7_audit(&phase7);
        phase7_shutdown(&phase7);
    }

    // Cleanup
    sandbox_cleanup(&sandbox);
    telemetry_close(&telemetry);

#if CUDA_ENABLED
    if (state.cuda_enabled) {
        qallow_cuda_cleanup(&state);
    }
#endif

    printf("\n[MAIN] Qallow VM shutdown complete\n");
    printf("[MAIN] Total runtime: %.2f ms\n", run_ms);
    return 0;
}