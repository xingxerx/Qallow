/**
 * Test harness for the recursive thinking system.
 * Demonstrates the memory feedback loop in action.
 */

#include "qallow/module.h"
#include "qallow/logging.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// External declarations for recursive thinking modules
extern ql_status mod_recursive_thinking_cycle(ql_state *S);
extern ql_status mod_store_thinking_output(ql_state *S);
extern ql_status mod_load_thinking_input(ql_state *S);
extern ql_status mod_extract_strategy_patterns(ql_state *S);
extern ql_status mod_generate_updated_strategy(ql_state *S);
extern ql_status mod_export_thinking_metrics(ql_state *S);

int main(int argc, char** argv) {
    int num_cycles = 5;
    
    if (argc > 1) {
        num_cycles = atoi(argv[1]);
        if (num_cycles < 1) num_cycles = 1;
        if (num_cycles > 20) num_cycles = 20;
    }
    
    qallow_logging_init();
    
    printf("╔════════════════════════════════════════════════════╗\n");
    printf("║  RECURSIVE THINKING TEST                           ║\n");
    printf("║  Memory Feedback Loop: Output → Memory → Input    ║\n");
    printf("╚════════════════════════════════════════════════════╝\n\n");
    
    printf("Running %d thinking cycles...\n", num_cycles);
    printf("Each cycle: Store Output → Load as Input → Learn Patterns → Update Strategy\n\n");
    
    // Initialize state
    ql_state state = {
        .t = 0.0,
        .reward = 0.5,
        .energy = 1.0,
        .risk = 0.3,
        .latent = NULL,
        .latent_bytes = 0
    };
    
    // Run multiple thinking cycles
    for (int cycle = 0; cycle < num_cycles; cycle++) {
        printf("\n");
        printf("═══════════════ Cycle %d ═══════════════\n", cycle + 1);
        
        // Simulate some variance in the state
        state.t = cycle * 10.0;
        state.reward += 0.05 * ((double)rand() / RAND_MAX - 0.5);
        state.energy *= 0.95 + 0.1 * ((double)rand() / RAND_MAX);
        state.risk = 0.2 + 0.3 * ((double)rand() / RAND_MAX);
        
        // Normalize
        state.reward = fmax(0.0, fmin(1.0, state.reward));
        state.energy = fmax(0.1, fmin(2.0, state.energy));
        state.risk = fmax(0.0, fmin(1.0, state.risk));
        
        printf("[Initial] t=%.1f, reward=%.3f, energy=%.3f, risk=%.3f\n",
               state.t, state.reward, state.energy, state.risk);
        
        // 1. Load previous thinking as input (if available)
        ql_status status = mod_load_thinking_input(&state);
        if (status.code == 0) {
            printf("✓ Loaded past wisdom into current thinking\n");
        }
        
        // 2. Do some "thinking" - simulate decision making
        printf("  → Processing current situation...\n");
        state.reward += 0.1;
        state.energy -= 0.05;
        
        // 3. Store the thinking output for future use
        status = mod_store_thinking_output(&state);
        if (status.code == 0) {
            printf("✓ Stored thinking episode to memory\n");
        }
        
        // 4. Try to extract patterns (needs enough episodes)
        status = mod_extract_strategy_patterns(&state);
        if (status.code == 0) {
            printf("✓ Extracted strategic patterns from past episodes\n");
        }
        
        // 5. Generate updated strategy based on patterns
        status = mod_generate_updated_strategy(&state);
        if (status.code == 0) {
            printf("✓ Generated updated strategy from learned patterns\n");
        }
        
        printf("[Updated] t=%.1f, reward=%.3f, energy=%.3f, risk=%.3f\n",
               state.t, state.reward, state.energy, state.risk);
        
        // 6. Export metrics
        status = mod_export_thinking_metrics(&state);
    }
    
    printf("\n\n");
    printf("╔════════════════════════════════════════════════════╗\n");
    printf("║  ✅ RECURSIVE THINKING TEST COMPLETE               ║\n");
    printf("╚════════════════════════════════════════════════════╝\n");
    printf("\nCheck data/logs/ for detailed telemetry.\n");
    
    return 0;
}
