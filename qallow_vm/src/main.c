#include "qallow_kernel.h"
#include "ppai.h"
#include "qcp.h"
#include "ethics.h"
#include "overlay.h"
#include "sandbox.h"

// Main application entry point for Qallow VM
// Supports both CUDA and CPU execution

static void print_banner(void) {
    printf("╔════════════════════════════════════════╗\n");
    printf("║     QALLOW VM - Unified System         ║\n");
    printf("║  Photonic & Quantum Hardware Emulation ║\n");
    printf("║  CPU + CUDA Acceleration Support       ║\n");
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

int main(void) {
    print_banner();
    
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
    
    print_system_info(&state);
    
    // Create initial safety snapshot
    sandbox_create_snapshot(&sandbox, &state, "initial_safe_state");
    
    printf("[MAIN] Starting VM execution loop...\n\n");
    
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
                return 1;
            }
        }
        
        // Print status every 100 ticks or on interesting changes
        if (tick % 100 == 0 || tick < 10) {
            qallow_print_status(&state, tick);
        }
        
        // Create periodic snapshots
        if (tick % 500 == 0 && tick > 0) {
            char snapshot_name[64];
            snprintf(snapshot_name, sizeof(snapshot_name), "auto_snapshot_tick_%d", tick);
            sandbox_create_snapshot(&sandbox, &state, snapshot_name);
        }
        
        // Early termination if system reaches stable equilibrium
        if (state.global_coherence > 0.999f && state.decoherence_level < 0.0001f) {
            printf("[KERNEL] System reached stable equilibrium at tick %d\n", tick);
            break;
        }
    }
    
    printf("\n[MAIN] VM execution completed\n");
    
    // Final reports
    printf("\n═══ FINAL STATUS REPORT ═══\n");
    qallow_print_status(&state, state.tick_count);
    
    printf("\n═══ ETHICS REPORT ═══\n");
    ethics_print_report(&ethics);
    
    printf("\n═══ SANDBOX REPORT ═══\n");
    sandbox_print_resource_report(&sandbox);
    sandbox_list_snapshots(&sandbox);
    
    // Cleanup
    sandbox_cleanup(&sandbox);
    
#if CUDA_ENABLED
    if (state.cuda_enabled) {
        qallow_cuda_cleanup(&state);
    }
#endif
    
    printf("\n[MAIN] Qallow VM shutdown complete\n");
    return 0;
}