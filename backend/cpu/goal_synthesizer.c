// Goal Synthesizer - Proactive goal generation with ethics gating

#include "phase7.h"
#include "ethics.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>

// ============================================================================
// INITIALIZATION
// ============================================================================

int gs_init(goal_synthesizer_t* gs) {
    if (!gs) return -1;
    
    memset(gs, 0, sizeof(goal_synthesizer_t));
    
    // Default weights for goal scoring
    gs->w_benefit = 1.0;
    gs->w_risk = 1.5;      // Risk penalty weighted higher
    gs->w_clarity = 0.5;
    gs->w_cost = 0.8;
    
    gs->goal_count = 0;
    gs->last_synthesis_ts = (uint64_t)time(NULL);
    
    return 0;
}

void gs_shutdown(goal_synthesizer_t* gs) {
    if (!gs) return;
    memset(gs, 0, sizeof(goal_synthesizer_t));
}

// ============================================================================
// GOAL GENERATION
// ============================================================================

static void generate_goal_id(char* out, int max_len) {
    uint64_t ts = (uint64_t)time(NULL);
    snprintf(out, max_len, "GOAL_%016llx", (unsigned long long)ts);
}

int gs_propose(goal_synthesizer_t* gs, const char* input_text, goal_t* out_goals, int max_goals) {
    if (!gs || !input_text || !out_goals || max_goals <= 0) return -1;
    if (gs->goal_count >= GS_MAX_GOALS) return -1;
    
    // Parse input text and synthesize goals
    // For now, create a simple demonstration goal
    int goals_generated = 0;
    
    if (strstr(input_text, "optimize") || strstr(input_text, "improve")) {
        goal_t* goal = &out_goals[goals_generated++];
        memset(goal, 0, sizeof(goal_t));
        
        generate_goal_id(goal->id, sizeof(goal->id));
        snprintf(goal->description, sizeof(goal->description), 
                "Optimize system performance based on: %.*s", 
                (int)(sizeof(goal->description) - 50), input_text);
        
        goal->priority = 0.7;
        goal->risk = 0.2;
        goal->clarity = 0.8;
        goal->cost = 0.4;
        goal->benefit = 0.75;
        goal->status = GOAL_STATUS_PROPOSED;
        goal->created_ts = (uint64_t)time(NULL);
        goal->constraint_count = 0;
        goal->smg_node_id = -1;
    }
    
    if (strstr(input_text, "learn") || strstr(input_text, "adapt")) {
        if (goals_generated < max_goals) {
            goal_t* goal = &out_goals[goals_generated++];
            memset(goal, 0, sizeof(goal_t));
            
            generate_goal_id(goal->id, sizeof(goal->id));
            snprintf(goal->description, sizeof(goal->description),
                    "Adapt system behavior through learning");
            
            goal->priority = 0.6;
            goal->risk = 0.3;
            goal->clarity = 0.7;
            goal->cost = 0.5;
            goal->benefit = 0.8;
            goal->status = GOAL_STATUS_PROPOSED;
            goal->created_ts = (uint64_t)time(NULL);
            goal->constraint_count = 0;
            goal->smg_node_id = -1;
        }
    }
    
    // Add goals to synthesizer
    for (int i = 0; i < goals_generated && gs->goal_count < GS_MAX_GOALS; i++) {
        memcpy(&gs->goals[gs->goal_count++], &out_goals[i], sizeof(goal_t));
    }
    
    gs->last_synthesis_ts = (uint64_t)time(NULL);
    return goals_generated;
}

int gs_propose_from_telemetry(goal_synthesizer_t* gs, const void* telemetry_data, 
                               goal_t* out_goals, int max_goals) {
    if (!gs || !telemetry_data || !out_goals || max_goals <= 0) return -1;
    
    // Analyze telemetry for goal opportunities
    // For demonstration, create goals based on system metrics
    int goals_generated = 0;
    
    // Example: Monitor coherence and generate stabilization goal if needed
    goal_t* goal = &out_goals[goals_generated++];
    memset(goal, 0, sizeof(goal_t));
    
    generate_goal_id(goal->id, sizeof(goal->id));
    snprintf(goal->description, sizeof(goal->description),
            "Maintain system coherence above threshold");
    
    goal->priority = 0.8;
    goal->risk = 0.15;
    goal->clarity = 0.9;
    goal->cost = 0.3;
    goal->benefit = 0.85;
    goal->status = GOAL_STATUS_PROPOSED;
    goal->created_ts = (uint64_t)time(NULL);
    goal->constraint_count = 1;
    snprintf(goal->constraints[0], sizeof(goal->constraints[0]), "coherence >= 0.99");
    goal->smg_node_id = -1;
    
    // Add to synthesizer
    for (int i = 0; i < goals_generated && gs->goal_count < GS_MAX_GOALS; i++) {
        memcpy(&gs->goals[gs->goal_count++], &out_goals[i], sizeof(goal_t));
    }
    
    return goals_generated;
}

// ============================================================================
// GOAL SCORING
// ============================================================================

double gs_score_goal(const goal_synthesizer_t* gs, const goal_t* goal) {
    if (!gs || !goal) return 0.0;
    
    // Priority = w1*Benefit - w2*Risk + w3*Clarity - w4*Cost
    double score = gs->w_benefit * goal->benefit
                 - gs->w_risk * goal->risk
                 + gs->w_clarity * goal->clarity
                 - gs->w_cost * goal->cost;
    
    return score;
}

static int compare_goals_by_score(const void* a, const void* b) {
    const goal_t* goal_a = (const goal_t*)a;
    const goal_t* goal_b = (const goal_t*)b;
    
    // Use priority field for comparison (should be pre-computed)
    if (goal_b->priority > goal_a->priority) return 1;
    if (goal_b->priority < goal_a->priority) return -1;
    return 0;
}

int gs_rank_goals(goal_synthesizer_t* gs) {
    if (!gs) return -1;
    
    // Compute scores for all proposed goals
    for (int i = 0; i < gs->goal_count; i++) {
        if (gs->goals[i].status == GOAL_STATUS_PROPOSED) {
            gs->goals[i].priority = gs_score_goal(gs, &gs->goals[i]);
        }
    }
    
    // Sort goals by priority descending
    qsort(gs->goals, gs->goal_count, sizeof(goal_t), compare_goals_by_score);
    
    return 0;
}

// ============================================================================
// GOAL COMMITMENT (WITH ETHICS GATE)
// ============================================================================

int gs_commit(goal_synthesizer_t* gs, const char* goal_id, const void* ethics_state) {
    if (!gs || !goal_id) return -1;
    
    // Find goal
    goal_t* goal = NULL;
    for (int i = 0; i < gs->goal_count; i++) {
        if (strcmp(gs->goals[i].id, goal_id) == 0) {
            goal = &gs->goals[i];
            break;
        }
    }
    
    if (!goal || goal->status != GOAL_STATUS_PROPOSED) return -1;
    
    // ETHICS GATE CHECK
    if (ethics_state) {
        const ethics_state_t* eth = (const ethics_state_t*)ethics_state;
        
        // Check total ethics score E = S + C + H
        // Using standard ethics_state_t structure
        double E = eth->safety_score + eth->clarity_score + eth->human_benefit_score;
        
        // Hard stop: E < 2.95
        if (E < 2.95) {
            goal->status = GOAL_STATUS_REJECTED;
            return -2;  // Ethics gate failure
        }
        
        // Risk threshold check
        if (goal->risk > 0.8) {
            goal->status = GOAL_STATUS_REJECTED;
            return -3;  // Risk too high
        }
    }
    
    // Pass ethics gate - commit goal
    goal->status = GOAL_STATUS_COMMITTED;
    goal->committed_ts = (uint64_t)time(NULL);
    
    return 0;
}

int gs_reject(goal_synthesizer_t* gs, const char* goal_id, const char* reason) {
    if (!gs || !goal_id) return -1;
    
    for (int i = 0; i < gs->goal_count; i++) {
        if (strcmp(gs->goals[i].id, goal_id) == 0) {
            gs->goals[i].status = GOAL_STATUS_REJECTED;
            return 0;
        }
    }
    
    return -1;
}

// ============================================================================
// GOAL LIFECYCLE
// ============================================================================

int gs_activate(goal_synthesizer_t* gs, const char* goal_id) {
    if (!gs || !goal_id) return -1;
    
    for (int i = 0; i < gs->goal_count; i++) {
        if (strcmp(gs->goals[i].id, goal_id) == 0) {
            if (gs->goals[i].status != GOAL_STATUS_COMMITTED) return -1;
            gs->goals[i].status = GOAL_STATUS_ACTIVE;
            return 0;
        }
    }
    
    return -1;
}

int gs_complete(goal_synthesizer_t* gs, const char* goal_id, double outcome_score) {
    if (!gs || !goal_id) return -1;
    
    for (int i = 0; i < gs->goal_count; i++) {
        if (strcmp(gs->goals[i].id, goal_id) == 0) {
            gs->goals[i].status = GOAL_STATUS_COMPLETED;
            gs->goals[i].benefit = outcome_score;  // Store actual outcome
            return 0;
        }
    }
    
    return -1;
}

int gs_fail(goal_synthesizer_t* gs, const char* goal_id, const char* reason) {
    if (!gs || !goal_id) return -1;
    
    for (int i = 0; i < gs->goal_count; i++) {
        if (strcmp(gs->goals[i].id, goal_id) == 0) {
            gs->goals[i].status = GOAL_STATUS_FAILED;
            return 0;
        }
    }
    
    return -1;
}

// ============================================================================
// QUERY
// ============================================================================

int gs_get_goal(const goal_synthesizer_t* gs, const char* goal_id, goal_t* out) {
    if (!gs || !goal_id || !out) return -1;
    
    for (int i = 0; i < gs->goal_count; i++) {
        if (strcmp(gs->goals[i].id, goal_id) == 0) {
            memcpy(out, &gs->goals[i], sizeof(goal_t));
            return 0;
        }
    }
    
    return -1;
}

int gs_list_goals(const goal_synthesizer_t* gs, goal_status_t status, 
                  goal_t* out_goals, int max_goals) {
    if (!gs || !out_goals || max_goals <= 0) return -1;
    
    int count = 0;
    for (int i = 0; i < gs->goal_count && count < max_goals; i++) {
        if (status == gs->goals[i].status || status == -1) {  // -1 = all
            memcpy(&out_goals[count++], &gs->goals[i], sizeof(goal_t));
        }
    }
    
    return count;
}
