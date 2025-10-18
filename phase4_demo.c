#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "qallow_kernel.h"
#include "multi_pocket.h"
#include "chronometric.h"
#include "ethics.h"

// Phase IV Demo: Multi-Pocket Simulation with Chronometric Prediction

void print_banner() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                   QALLOW PHASE IV DEMO                       ║\n");
    printf("║                                                              ║\n");
    printf("║  Multi-Pocket Simulation Scheduler                           ║\n");
    printf("║  Chronometric Prediction Layer                               ║\n");
    printf("║  Temporal Time Bank & Drift Detection                        ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    printf("\n");
}

void print_phase_separator(const char* phase_name) {
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  %s\n", phase_name);
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("\n");
}

int main(int argc, char** argv) {
    srand((unsigned int)time(NULL));
    
    print_banner();
    
    // Configuration
    int num_pockets = 8;
    int num_ticks = 100;
    
    if (argc > 1) num_pockets = atoi(argv[1]);
    if (argc > 2) num_ticks = atoi(argv[2]);
    
    if (num_pockets > MAX_POCKETS) num_pockets = MAX_POCKETS;
    
    printf("Configuration:\n");
    printf("  Number of pockets: %d\n", num_pockets);
    printf("  Simulation ticks:  %d\n", num_ticks);
    printf("\n");
    
    // ===================================================================
    // PHASE 1: Initialize Main Qallow State
    // ===================================================================
    
    print_phase_separator("PHASE 1: Initialize Main Qallow VM");
    
    qallow_state_t main_state;
    qallow_kernel_init(&main_state, 256, 3);
    
    printf("Main VM initialized:\n");
    printf("  Nodes: %d\n", 256);
    printf("  Overlays: %d\n", 3);
    printf("  Initial coherence: %.4f\n", main_state.global_coherence);
    
    // ===================================================================
    // PHASE 2: Initialize Multi-Pocket Scheduler
    // ===================================================================
    
    print_phase_separator("PHASE 2: Initialize Multi-Pocket Scheduler");
    
    multi_pocket_scheduler_t scheduler;
    multi_pocket_init(&scheduler, num_pockets);
    
    // Generate random parameters for each pocket
    multi_pocket_generate_random_params(&scheduler);
    
    // ===================================================================
    // PHASE 3: Initialize Chronometric Prediction Layer
    // ===================================================================
    
    print_phase_separator("PHASE 3: Initialize Chronometric Prediction");
    
    chronometric_state_t chrono;
    chronometric_init(&chrono, 0.01f, 0.95f);
    
    // ===================================================================
    // PHASE 4: Run Multi-Pocket Simulation
    // ===================================================================
    
    print_phase_separator("PHASE 4: Execute Multi-Pocket Simulation");
    
    clock_t sim_start = clock();
    
    multi_pocket_execute_all(&scheduler, &main_state, num_ticks);
    
    clock_t sim_end = clock();
    double sim_time = (double)(sim_end - sim_start) / CLOCKS_PER_SEC;
    
    printf("\nSimulation complete in %.2f seconds\n", sim_time);
    
    // ===================================================================
    // PHASE 5: Analyze Results
    // ===================================================================
    
    print_phase_separator("PHASE 5: Analyze Pocket Results");
    
    multi_pocket_print_results(&scheduler);
    multi_pocket_print_statistics(&scheduler);
    
    // Calculate consensus
    float consensus = multi_pocket_calculate_consensus(&scheduler);
    printf("Pocket Consensus: %.4f\n", consensus);
    
    // ===================================================================
    // PHASE 6: Merge Pockets into Main State
    // ===================================================================
    
    print_phase_separator("PHASE 6: Merge Pocket Worldlines");
    
    pocket_merge_config_t merge_config;
    merge_config.use_weighted_merge = true;
    merge_config.filter_outliers = true;
    merge_config.outlier_threshold = 0.1f;
    merge_config.confidence_weight = 2.0f;
    
    qallow_state_t merged_state;
    multi_pocket_merge(&scheduler, &merged_state, &merge_config);
    
    printf("Merged state metrics:\n");
    printf("  Global coherence:  %.4f\n", merged_state.global_coherence);
    printf("  Decoherence level: %.6f\n", merged_state.decoherence_level);
    
    // ===================================================================
    // PHASE 7: Chronometric Prediction
    // ===================================================================
    
    print_phase_separator("PHASE 7: Chronometric Prediction & Time Bank");
    
    // Simulate temporal observations
    double base_time = 0.0;
    for (int tick = 0; tick < num_ticks; tick += 10) {
        double observed_time = base_time + tick * 0.1 + ((rand() % 100) - 50) * 0.0001;
        double predicted_time = base_time + tick * 0.1;
        
        // Get pocket result for this tick range
        int pocket_idx = tick % scheduler.num_pockets;
        qallow_state_t* pocket_state = &scheduler.results[pocket_idx].final_state;
        
        chronometric_update(&chrono, tick, observed_time, predicted_time, pocket_state);
    }
    
    // Print time bank statistics
    chrono_bank_print_stats(&chrono.time_bank);
    
    // Generate forecast
    chronometric_generate_forecast(&chrono, num_ticks, &merged_state);
    chronometric_print_forecast(&chrono, 20);
    
    // Analyze temporal patterns
    chronometric_analyze_patterns(&chrono);
    
    // ===================================================================
    // PHASE 8: Ethics Check
    // ===================================================================
    
    print_phase_separator("PHASE 8: Ethics & Safety Verification");
    
    ethics_state_t ethics;
    qallow_ethics_check(&merged_state, &ethics);
    
    printf("Ethics Scores:\n");
    printf("  Safety (S):      %.4f / 1.0\n", ethics.safety_score);
    printf("  Consistency (C): %.4f / 1.0\n", ethics.consistency_score);
    printf("  Harmony (H):     %.4f / 1.0\n", ethics.harmony_score);
    printf("  Total Ethics:    %.4f / 3.0\n", ethics.total_ethics_score);
    
    bool ethics_ok = ethics.total_ethics_score >= 2.5f;
    printf("\nEthics Check: %s\n", ethics_ok ? "✓ PASS" : "✗ FAIL");
    
    // ===================================================================
    // PHASE 9: Write Summary Reports
    // ===================================================================
    
    print_phase_separator("PHASE 9: Generate Summary Reports");
    
    multi_pocket_write_summary(&scheduler);
    chronometric_write_summary(&chrono);
    
    printf("Reports generated:\n");
    printf("  - multi_pocket_summary.txt\n");
    printf("  - chronometric_summary.txt\n");
    printf("  - qallow_multi_pocket.csv\n");
    printf("  - chronometric_telemetry.csv\n");
    printf("  - pocket_[0-%d].csv (per-pocket telemetry)\n", num_pockets - 1);
    
    // ===================================================================
    // PHASE 10: Cleanup
    // ===================================================================
    
    print_phase_separator("PHASE 10: Cleanup & Shutdown");
    
    multi_pocket_cleanup(&scheduler);
    chronometric_cleanup(&chrono);
    
    printf("Cleanup complete.\n");
    
    // ===================================================================
    // Final Summary
    // ===================================================================
    
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                     EXECUTION SUMMARY                        ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║  Pockets Simulated:    %-6d                              ║\n", num_pockets);
    printf("║  Ticks per Pocket:     %-6d                              ║\n", num_ticks);
    printf("║  Total Simulation Time: %-6.2f sec                        ║\n", sim_time);
    printf("║  Merged Coherence:     %-6.4f                            ║\n", merged_state.global_coherence);
    printf("║  Pocket Consensus:     %-6.4f                            ║\n", consensus);
    printf("║  Ethics Score:         %-6.4f / 3.0                      ║\n", ethics.total_ethics_score);
    printf("║  Temporal Drift:       %-8.6f sec                       ║\n", chrono.accumulated_drift);
    printf("║  Prediction Conf:      %-6.4f                            ║\n", chrono_bank_get_confidence(&chrono.time_bank));
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║  Phase IV Status:      %-35s   ║\n", 
           (consensus > 0.8f && ethics_ok) ? "✓ SUCCESS" : "✗ NEEDS TUNING");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    printf("\n");
    
    return 0;
}
