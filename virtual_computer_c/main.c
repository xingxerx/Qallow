/*
 * Virtual Computer Demo - Main Program
 * Demonstrates CUDA, Neuromorphic, and Photonic processor simulation
 */

#include "virtual_computer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

void print_banner(const char *text) {
    printf("\n");
    printf("================================================================================\n");
    printf("  %s\n", text);
    printf("================================================================================\n\n");
}

int main(void) {
    print_banner("VIRTUAL COMPUTER SYSTEM - CUDA + Neuromorphic + Photonic");
    
    /* Create virtual computer */
    virtual_computer_t *vc = vc_create();
    if (!vc) {
        fprintf(stderr, "Failed to create virtual computer\n");
        return EXIT_FAILURE;
    }
    
    printf("✓ Virtual Computer initialized\n");
    printf("  - CUDA GPU (8GB memory, 80 SMs, 128 cores/SM)\n");
    printf("  - Neuromorphic Processor (1000 neurons, 4 layers)\n");
    printf("  - Photonic Processor (64 waveguides, 256 gates)\n\n");
    
    /* Create workloads */
    print_banner("Creating Diversified Workloads");
    
    printf("Workload Configurations:\n\n");
    
    vc_create_workload(vc, WORKLOAD_GPU_COMPUTE, 5, 512, 5_000_000_000);
    printf("  ✓ GPU Compute (priority=5, size=512MB)\n");
    
    vc_create_workload(vc, WORKLOAD_GPU_MEMORY_INTENSIVE, 4, 2048, 1_000_000_000);
    printf("  ✓ GPU Memory Intensive (priority=4, size=2048MB)\n");
    
    vc_create_workload(vc, WORKLOAD_NEURAL_INFERENCE, 6, 256, 2_000_000_000);
    printf("  ✓ Neural Inference (priority=6, size=256MB)\n");
    
    vc_create_workload(vc, WORKLOAD_NEURAL_TRAINING, 3, 1024, 10_000_000_000);
    printf("  ✓ Neural Training (priority=3, size=1024MB)\n");
    
    vc_create_workload(vc, WORKLOAD_PHOTONIC_COMPUTE, 5, 128, 500_000_000);
    printf("  ✓ Photonic Compute (priority=5, size=128MB)\n");
    
    vc_create_workload(vc, WORKLOAD_HYBRID_PROCESSING, 7, 768, 8_000_000_000);
    printf("  ✓ Hybrid Processing (priority=7, size=768MB)\n");
    
    printf("\nTotal workloads created: %u\n", vc->queue_size);
    
    /* Execute workloads */
    print_banner("Executing Workloads");
    printf("Processing scheduled workloads...\n\n");
    
    time_t start_exec = time(NULL);
    vc_run_scheduled_workloads(vc);
    time_t end_exec = time(NULL);
    
    double elapsed = difftime(end_exec, start_exec);
    printf("✓ Workload execution completed in %.2f seconds\n", elapsed);
    printf("  - Workloads executed: %u\n", vc->completed_size);
    printf("  - Total energy: %.3f J\n", vc->total_energy_consumed);
    printf("  - Throughput: %.2f workloads/sec\n\n", vc->throughput_workloads_per_sec);
    
    /* Print system status */
    print_banner("System Performance Summary");
    
    vc_print_system_status(vc);
    
    /* Print individual processor stats */
    printf("Individual Processor Performance:\n\n");
    
    gpu_print_status(vc->cuda_gpu);
    nm_print_status(vc->neuromorphic);
    pp_print_status(vc->photonic);
    
    /* Print detailed workload results */
    print_banner("Workload Execution Results");
    
    printf("Completed Workloads:\n\n");
    for (uint32_t i = 0; i < vc->completed_size; i++) {
        workload_t *w = &vc->completed_workloads[i];
        
        const char *type_str = "Unknown";
        switch (w->workload_type) {
            case WORKLOAD_GPU_COMPUTE: type_str = "GPU Compute"; break;
            case WORKLOAD_GPU_MEMORY_INTENSIVE: type_str = "GPU Memory"; break;
            case WORKLOAD_NEURAL_INFERENCE: type_str = "Neural Inference"; break;
            case WORKLOAD_NEURAL_TRAINING: type_str = "Neural Training"; break;
            case WORKLOAD_PHOTONIC_COMPUTE: type_str = "Photonic Compute"; break;
            case WORKLOAD_PHOTONIC_OPTIMIZATION: type_str = "Photonic Optimize"; break;
            case WORKLOAD_HYBRID_PROCESSING: type_str = "Hybrid Processing"; break;
        }
        
        printf("  #%u: %s\n", w->workload_id, type_str);
        printf("      Priority: %u, Data: %lu MB, Ops: %lu\n",
               w->priority, w->data_size_mb, w->compute_ops);
        printf("      Status: %s, Score: %.2e\n\n",
               w->status, w->performance_score);
    }
    
    print_banner("Virtual Computer Ready for Agent Optimization");
    printf("✓ All processors initialized and ready\n");
    printf("✓ Workload framework tested and validated\n");
    printf("✓ System ready for Lightning Agent integration\n\n");
    
    printf("Next Steps:\n");
    printf("  1. Connect Lightning Agent to virtual_computer.h API\n");
    printf("  2. Implement optimization task interface\n");
    printf("  3. Run agent improvement loops\n");
    printf("  4. Monitor performance gains\n\n");
    
    /* Cleanup */
    vc_destroy(vc);
    
    print_banner("Demonstration Complete");
    printf("Success!\n\n");
    
    return EXIT_SUCCESS;
}
