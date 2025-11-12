/**
 * Recursive Memory-Based Thinking Module
 * 
 * This module implements a feedback loop where the AGI:
 * 1. Stores its thinking outputs in memory
 * 2. Retrieves previous thinking patterns
 * 3. Uses them as input for new strategic thinking
 * 4. Generates updated/improved strategies
 * 
 * This creates a self-improving cognitive loop.
 */

#include "qallow/module.h"
#include "qallow/logging.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#define MAX_THINKING_EPISODES 256
#define MAX_STRATEGY_PATTERNS 64
#define THINKING_DIM 16

// Represents a single thinking episode (output that becomes future input)
typedef struct {
    double strategy_vector[THINKING_DIM];  // Abstract strategy representation
    double effectiveness;                   // How well did this strategy work?
    double context_state[8];                // State when this strategy was used
    double timestamp;
    char strategy_tag[64];                  // Human-readable tag
    int generation;                         // Which iteration produced this
} thinking_episode_t;

// Represents learned strategy patterns
typedef struct {
    double pattern[THINKING_DIM];
    double success_rate;
    double usage_count;
    double evolution_factor;                // How fast this pattern evolves
    char pattern_name[64];
} strategy_pattern_t;

// Global memory for thinking episodes and patterns
static thinking_episode_t thinking_memory[MAX_THINKING_EPISODES] = {0};
static int thinking_count = 0;
static strategy_pattern_t learned_strategies[MAX_STRATEGY_PATTERNS] = {0};
static int strategy_count = 0;
static int current_generation = 0;

// Internal state for recursive thinking
static double accumulated_wisdom = 0.0;
static double strategy_diversity = 1.0;
static double confidence_in_patterns = 0.0;

/**
 * Store current thinking output into memory for future use
 */
ql_status mod_store_thinking_output(ql_state *S) {
    if (thinking_count >= MAX_THINKING_EPISODES) {
        // Memory full - consolidate (keep best 50%)
        qallow_log_info("RECURSIVE_THINKING", 
                       "Memory full, consolidating thinking episodes");
        
        // Sort by effectiveness
        for (int i = 0; i < thinking_count - 1; i++) {
            for (int j = i + 1; j < thinking_count; j++) {
                if (thinking_memory[i].effectiveness < thinking_memory[j].effectiveness) {
                    thinking_episode_t tmp = thinking_memory[i];
                    thinking_memory[i] = thinking_memory[j];
                    thinking_memory[j] = tmp;
                }
            }
        }
        thinking_count = thinking_count / 2;
    }

    // Create new thinking episode from current state
    thinking_episode_t new_episode = {0};

    // Encode current state into strategy vector (using only available ql_state fields)
    new_episode.strategy_vector[0] = S->energy;
    new_episode.strategy_vector[1] = S->risk;
    new_episode.strategy_vector[2] = S->reward;
    new_episode.strategy_vector[3] = S->t / 1000.0;  // Normalized time
    new_episode.strategy_vector[4] = accumulated_wisdom;
    new_episode.strategy_vector[5] = strategy_diversity;
    new_episode.strategy_vector[6] = confidence_in_patterns;
    new_episode.strategy_vector[7] = sin(S->t / 100.0);  // Temporal pattern

    // Add more strategic dimensions
    new_episode.strategy_vector[8] = cos(S->t / 100.0);
    new_episode.strategy_vector[9] = S->energy * S->risk;  // Energy-risk product
    new_episode.strategy_vector[10] = S->reward / (1.0 + S->risk);  // Risk-adjusted reward
    new_episode.strategy_vector[11] = fabs(S->energy - 0.5);  // Energy deviation
    new_episode.strategy_vector[12] = fabs(S->risk - 0.5);  // Risk deviation
    new_episode.strategy_vector[13] = S->reward * (1.0 - S->risk);  // Reward-risk balance
    new_episode.strategy_vector[14] = accumulated_wisdom * strategy_diversity;
    new_episode.strategy_vector[15] = confidence_in_patterns * (1.0 - S->risk);

    // Store context (using only available fields)
    new_episode.context_state[0] = S->energy;
    new_episode.context_state[1] = S->risk;
    new_episode.context_state[2] = S->reward;
    new_episode.context_state[3] = S->t / 1000.0;
    new_episode.context_state[4] = accumulated_wisdom;
    new_episode.context_state[5] = strategy_diversity;
    new_episode.context_state[6] = confidence_in_patterns;
    new_episode.context_state[7] = sin(S->t / 100.0);

    // Calculate effectiveness (how well is the system doing?)
    new_episode.effectiveness =
        0.3 * S->energy +
        0.3 * S->reward +
        0.2 * (1.0 - S->risk) +
        0.1 * accumulated_wisdom +
        0.1 * strategy_diversity;
    
    new_episode.timestamp = S->t;
    new_episode.generation = current_generation;
    
    snprintf(new_episode.strategy_tag, sizeof(new_episode.strategy_tag),
             "gen%d_t%.0f_eff%.2f", current_generation, S->t, new_episode.effectiveness);
    
    thinking_memory[thinking_count++] = new_episode;
    
    return (ql_status){0, "thinking output stored"};
}

/**
 * Load previous thinking patterns and use them to inform current strategy
 */
ql_status mod_load_thinking_input(ql_state *S) {
    if (thinking_count < 3) {
        return (ql_status){0, "insufficient thinking history"};
    }
    
    // Find the most effective past thinking episodes
    double best_effectiveness = 0.0;
    int best_idx = 0;
    
    for (int i = 0; i < thinking_count; i++) {
        // Calculate similarity to current context
        double context_similarity = 0.0;
        double sum_sq_diff = 0.0;
        
        for (int j = 0; j < 8; j++) {
            double diff = S->energy - thinking_memory[i].context_state[j];  // Simplified
            sum_sq_diff += diff * diff;
        }
        
        context_similarity = exp(-sum_sq_diff / 8.0);  // Gaussian similarity
        
        // Weight by both effectiveness and context similarity
        double relevance = thinking_memory[i].effectiveness * context_similarity;
        
        if (relevance > best_effectiveness) {
            best_effectiveness = relevance;
            best_idx = i;
        }
    }
    
    // Apply the best past strategy to current state (feedback loop)
    double blend_factor = 0.15;  // How much to blend past wisdom
    
    S->energy = (1.0 - blend_factor) * S->energy +
                blend_factor * thinking_memory[best_idx].strategy_vector[0];
    S->risk = (1.0 - blend_factor) * S->risk +
              blend_factor * thinking_memory[best_idx].strategy_vector[1];
    S->reward = (1.0 - blend_factor) * S->reward +
                blend_factor * thinking_memory[best_idx].strategy_vector[2];
    
    // Update accumulated wisdom
    accumulated_wisdom = 0.95 * accumulated_wisdom + 0.05 * thinking_memory[best_idx].effectiveness;
    
    qallow_log_info("RECURSIVE_THINKING", 
                   "Loaded past strategy: %s (effectiveness: %.3f)",
                   thinking_memory[best_idx].strategy_tag,
                   thinking_memory[best_idx].effectiveness);
    
    return (ql_status){0, "thinking input loaded"};
}

/**
 * Extract learned patterns from thinking episodes
 */
ql_status mod_extract_strategy_patterns(ql_state *S) {
    if (thinking_count < 10) {
        return (ql_status){0, "insufficient data for pattern extraction"};
    }
    
    // Cluster thinking episodes into patterns
    for (int i = 0; i < thinking_count && strategy_count < MAX_STRATEGY_PATTERNS; i++) {
        if (thinking_memory[i].effectiveness < 0.5) continue;  // Only learn from good episodes
        
        // Check if this matches an existing pattern
        int matched_pattern = -1;
        double min_distance = 1e9;
        
        for (int p = 0; p < strategy_count; p++) {
            double distance = 0.0;
            for (int d = 0; d < THINKING_DIM; d++) {
                double diff = thinking_memory[i].strategy_vector[d] - learned_strategies[p].pattern[d];
                distance += diff * diff;
            }
            distance = sqrt(distance);
            
            if (distance < min_distance) {
                min_distance = distance;
                matched_pattern = p;
            }
        }
        
        // If close enough to existing pattern, update it
        if (matched_pattern >= 0 && min_distance < 0.5) {
            double alpha = 0.1;  // Learning rate
            for (int d = 0; d < THINKING_DIM; d++) {
                learned_strategies[matched_pattern].pattern[d] = 
                    (1.0 - alpha) * learned_strategies[matched_pattern].pattern[d] +
                    alpha * thinking_memory[i].strategy_vector[d];
            }
            learned_strategies[matched_pattern].usage_count += 1.0;
            learned_strategies[matched_pattern].success_rate = 
                0.9 * learned_strategies[matched_pattern].success_rate + 
                0.1 * thinking_memory[i].effectiveness;
        }
        // Otherwise create new pattern
        else if (strategy_count < MAX_STRATEGY_PATTERNS) {
            memcpy(learned_strategies[strategy_count].pattern, 
                   thinking_memory[i].strategy_vector,
                   sizeof(double) * THINKING_DIM);
            learned_strategies[strategy_count].success_rate = thinking_memory[i].effectiveness;
            learned_strategies[strategy_count].usage_count = 1.0;
            learned_strategies[strategy_count].evolution_factor = 0.05;
            snprintf(learned_strategies[strategy_count].pattern_name,
                    sizeof(learned_strategies[strategy_count].pattern_name),
                    "pattern_%d_gen%d", strategy_count, current_generation);
            strategy_count++;
        }
    }
    
    // Update confidence in patterns
    confidence_in_patterns = fmin(1.0, strategy_count / 10.0);
    
    return (ql_status){0, "strategy patterns extracted"};
}

/**
 * Generate updated strategy by combining learned patterns
 */
ql_status mod_generate_updated_strategy(ql_state *S) {
    if (strategy_count == 0) {
        return (ql_status){0, "no learned strategies available"};
    }
    
    // Find best strategies based on success rate and usage
    double best_score = 0.0;
    int best_strategy_idx = 0;
    
    for (int i = 0; i < strategy_count; i++) {
        double score = learned_strategies[i].success_rate * 
                      log(1.0 + learned_strategies[i].usage_count);
        if (score > best_score) {
            best_score = score;
            best_strategy_idx = i;
        }
    }
    
    // Apply the best learned strategy
    double application_strength = 0.2 * confidence_in_patterns;  // Scale by confidence
    
    S->energy = (1.0 - application_strength) * S->energy +
                application_strength * learned_strategies[best_strategy_idx].pattern[0];
    S->risk = (1.0 - application_strength) * S->risk +
              application_strength * learned_strategies[best_strategy_idx].pattern[1];
    S->reward = (1.0 - application_strength) * S->reward +
                application_strength * learned_strategies[best_strategy_idx].pattern[2];

    // Evolve the strategy slightly for exploration
    double evolution = learned_strategies[best_strategy_idx].evolution_factor;
    double exploration_noise = evolution * (((double)rand() / RAND_MAX) - 0.5);
    exploration_noise = fmax(0.0, fmin(1.0, exploration_noise));

    // Track diversity
    strategy_diversity = 0.98 * strategy_diversity + 0.02 * exploration_noise;
    
    // Increment generation
    current_generation++;
    
    qallow_log_info("RECURSIVE_THINKING",
                   "Applied strategy: %s (gen: %d, confidence: %.3f)",
                   learned_strategies[best_strategy_idx].pattern_name,
                   current_generation,
                   confidence_in_patterns);
    
    return (ql_status){0, "updated strategy generated"};
}

/**
 * Main recursive thinking cycle: Output -> Memory -> Input -> New Strategy
 */
ql_status mod_recursive_thinking_cycle(ql_state *S) {
    // Step 1: Store current thinking as output
    mod_store_thinking_output(S);
    
    // Step 2: Load relevant past thinking as input
    mod_load_thinking_input(S);
    
    // Step 3: Extract patterns from accumulated thinking
    if ((int)S->t % 50 == 0) {  // Periodic pattern extraction
        mod_extract_strategy_patterns(S);
    }
    
    // Step 4: Generate and apply updated strategy
    if (confidence_in_patterns > 0.3) {  // Only when we have enough learned patterns
        mod_generate_updated_strategy(S);
    }
    
    return (ql_status){0, "recursive thinking cycle complete"};
}

/**
 * Export thinking metrics for analysis
 */
ql_status mod_export_thinking_metrics(ql_state *S) {
    (void)S;  // Unused for now
    
    // Calculate some aggregate metrics
    double avg_effectiveness = 0.0;
    if (thinking_count > 0) {
        for (int i = 0; i < thinking_count; i++) {
            avg_effectiveness += thinking_memory[i].effectiveness;
        }
        avg_effectiveness /= thinking_count;
    }
    
    qallow_log_info("RECURSIVE_THINKING",
                   "Metrics - Episodes: %d, Patterns: %d, Avg Effectiveness: %.3f, "
                   "Wisdom: %.3f, Confidence: %.3f, Generation: %d",
                   thinking_count, strategy_count, avg_effectiveness,
                   accumulated_wisdom, confidence_in_patterns, current_generation);
    
    return (ql_status){0, "thinking metrics exported"};
}
