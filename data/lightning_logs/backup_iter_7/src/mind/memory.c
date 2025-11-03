#include "qallow/module.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

// Episodic memory: stores significant events
typedef struct {
    double energy;
    double risk;
    double reward;
    double timestamp;
    double significance;  // How important was this event?
} memory_episode_t;

#define MAX_EPISODES 1000
static memory_episode_t episodes[MAX_EPISODES] = {0};
static int episode_count = 0;

// Semantic memory: abstract patterns
typedef struct {
    double pattern[8];    // Abstract pattern vector
    double frequency;     // How often seen?
    double utility;       // How useful?
} memory_pattern_t;

#define MAX_PATTERNS 100
static memory_pattern_t patterns[MAX_PATTERNS] = {0};
static int pattern_count = 0;

// Store significant events in episodic memory
ql_status mod_episodic_memory(ql_state *S) {
    // Compute significance: how different from average?
    static double avg_reward = 0.0;
    avg_reward = 0.99 * avg_reward + 0.01 * S->reward;
    
    double significance = fabs(S->reward - avg_reward);
    
    // Store if significant
    if (significance > 0.05 && episode_count < MAX_EPISODES) {
        episodes[episode_count].energy = S->energy;
        episodes[episode_count].risk = S->risk;
        episodes[episode_count].reward = S->reward;
        episodes[episode_count].timestamp = S->t;
        episodes[episode_count].significance = significance;
        episode_count++;
    }
    
    return (ql_status){0, "episodic memory ok"};
}

// Extract and consolidate patterns from episodes
ql_status mod_semantic_memory(ql_state *S) {
    // Consolidate: find recurring patterns
    if (episode_count < 10) {
        return (ql_status){0, "semantic memory ok"};
    }
    
    // Cluster episodes by similarity
    for (int i = 0; i < episode_count && pattern_count < MAX_PATTERNS; i++) {
        double pattern[8] = {0};
        pattern[0] = episodes[i].energy;
        pattern[1] = episodes[i].risk;
        pattern[2] = episodes[i].reward;
        pattern[3] = episodes[i].significance;
        
        // Check if pattern already exists
        int found = 0;
        for (int p = 0; p < pattern_count; p++) {
            double dist = 0.0;
            for (int j = 0; j < 4; j++) {
                dist += fabs(pattern[j] - patterns[p].pattern[j]);
            }
            
            if (dist < 0.1) {  // Similar pattern
                patterns[p].frequency += 1.0;
                found = 1;
                break;
            }
        }
        
        if (!found && pattern_count < MAX_PATTERNS) {
            memcpy(patterns[pattern_count].pattern, pattern, sizeof(pattern));
            patterns[pattern_count].frequency = 1.0;
            patterns[pattern_count].utility = episodes[i].significance;
            pattern_count++;
        }
    }
    
    return (ql_status){0, "semantic memory ok"};
}

// Recall and apply learned patterns
ql_status mod_memory_recall(ql_state *S) {
    if (pattern_count == 0) {
        return (ql_status){0, "memory recall ok"};
    }
    
    // Find most useful pattern
    double best_utility = 0.0;
    int best_idx = 0;
    for (int p = 0; p < pattern_count; p++) {
        double utility = patterns[p].utility * patterns[p].frequency;
        if (utility > best_utility) {
            best_utility = utility;
            best_idx = p;
        }
    }
    
    // Blend current state with best pattern
    double blend = 0.1;  // 10% pattern influence
    S->energy = (1.0 - blend) * S->energy + blend * patterns[best_idx].pattern[0];
    S->risk = (1.0 - blend) * S->risk + blend * patterns[best_idx].pattern[1];
    S->reward = (1.0 - blend) * S->reward + blend * patterns[best_idx].pattern[2];
    
    return (ql_status){0, "memory recall ok"};
}

// Consolidate memory: compress old episodes
ql_status mod_memory_consolidation(ql_state *S) {
    // Periodically consolidate: keep only high-significance episodes
    if ((int)S->t % 100 != 0 || episode_count < 500) {
        return (ql_status){0, "consolidation ok"};
    }
    
    // Sort by significance
    for (int i = 0; i < episode_count - 1; i++) {
        for (int j = i + 1; j < episode_count; j++) {
            if (episodes[i].significance < episodes[j].significance) {
                memory_episode_t tmp = episodes[i];
                episodes[i] = episodes[j];
                episodes[j] = tmp;
            }
        }
    }
    
    // Keep top 50% most significant
    episode_count = episode_count / 2;
    
    return (ql_status){0, "consolidation ok"};
}

