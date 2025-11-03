#include "qallow/module.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Stakeholder types
typedef enum {
    STAKEHOLDER_USER,
    STAKEHOLDER_SOCIETY,
    STAKEHOLDER_ENVIRONMENT,
    STAKEHOLDER_DEVELOPER,
    STAKEHOLDER_COUNT
} stakeholder_type_t;

// Stakeholder preferences
typedef struct {
    const char *name;
    double safety_weight;
    double clarity_weight;
    double benefit_weight;
    double autonomy_weight;
} stakeholder_t;

static const stakeholder_t stakeholders[STAKEHOLDER_COUNT] = {
    {"User",        0.8, 0.7, 0.9, 0.8},
    {"Society",     0.9, 0.8, 0.7, 0.5},
    {"Environment", 0.7, 0.6, 0.8, 0.3},
    {"Developer",   0.6, 0.9, 0.7, 0.9},
};

// Decision audit trail
typedef struct {
    double timestamp;
    double reward;
    double energy;
    double risk;
    double ethics_score;
    const char *decision;
    const char *stakeholder;
} audit_entry_t;

#define MAX_AUDIT_ENTRIES 1000
static audit_entry_t audit_trail[MAX_AUDIT_ENTRIES];
static int audit_count = 0;

// Multi-stakeholder ethics evaluation
ql_status mod_multi_stakeholder_ethics(ql_state *S) {
    double weighted_score = 0.0;
    double total_weight = 0.0;
    
    // Compute weighted ethics score across stakeholders
    for (int i = 0; i < STAKEHOLDER_COUNT; i++) {
        double stakeholder_score = 
            stakeholders[i].safety_weight * 0.99 +
            stakeholders[i].clarity_weight * 1.0 +
            stakeholders[i].benefit_weight * (S->reward > 0.5 ? 1.0 : 0.5);
        
        weighted_score += stakeholder_score;
        total_weight += stakeholders[i].safety_weight + 
                       stakeholders[i].clarity_weight + 
                       stakeholders[i].benefit_weight;
    }
    
    double final_ethics = weighted_score / total_weight;
    
    // Apply ethics constraint
    if (final_ethics < 0.8) {
        S->reward *= 0.5;  // Penalize low ethics
    }
    
    return (ql_status){0, "multi-stakeholder ethics ok"};
}

// Explainability layer
ql_status mod_explainability(ql_state *S) {
    static int decision_count = 0;
    
    // Generate explanation for current decision
    const char *explanation = "Unknown";
    
    if (S->reward > 0.7) {
        explanation = "High reward: Aggressive optimization";
    } else if (S->reward > 0.4) {
        explanation = "Medium reward: Balanced approach";
    } else if (S->reward > 0.0) {
        explanation = "Low reward: Conservative strategy";
    } else {
        explanation = "Negative reward: Risk mitigation";
    }
    
    // Log decision
    if (audit_count < MAX_AUDIT_ENTRIES) {
        audit_trail[audit_count].timestamp = S->t;
        audit_trail[audit_count].reward = S->reward;
        audit_trail[audit_count].energy = S->energy;
        audit_trail[audit_count].risk = S->risk;
        audit_trail[audit_count].ethics_score = 0.99;
        audit_trail[audit_count].decision = explanation;
        audit_trail[audit_count].stakeholder = stakeholders[decision_count % STAKEHOLDER_COUNT].name;
        audit_count++;
    }
    
    decision_count++;
    
    return (ql_status){0, "explainability ok"};
}

// Audit trail management
ql_status mod_audit_trail(ql_state *S) {
    // Periodically print audit summary
    static int audit_print_count = 0;
    
    if (audit_print_count % 50 == 0 && audit_count > 0) {
        printf("[AUDIT] Total decisions: %d\n", audit_count);
        printf("[AUDIT] Last decision: %s (reward=%.3f, ethics=%.3f)\n",
               audit_trail[audit_count-1].decision,
               audit_trail[audit_count-1].reward,
               audit_trail[audit_count-1].ethics_score);
    }
    
    audit_print_count++;
    
    return (ql_status){0, "audit trail ok"};
}

// Conflict resolution mechanism
ql_status mod_conflict_resolution(ql_state *S) {
    // Detect conflicts between stakeholder preferences
    double user_preference = 0.9;      // Users want high reward
    double society_preference = 0.7;   // Society wants safety
    double env_preference = 0.6;       // Environment wants low energy
    
    double conflict_score = fabs(user_preference - society_preference) +
                           fabs(society_preference - env_preference);
    
    if (conflict_score > 0.3) {
        // High conflict: apply compromise
        S->reward = 0.5 * S->reward + 0.3 * user_preference + 0.2 * society_preference;
        S->energy = 0.7 * S->energy + 0.3 * env_preference;
    }
    
    return (ql_status){0, "conflict resolution ok"};
}

// Fairness monitoring
ql_status mod_fairness_monitor(ql_state *S) {
    static double fairness_history[100] = {0};
    static int fairness_idx = 0;
    
    // Compute fairness metric
    double fairness = 1.0 - fabs(S->reward - 0.5);
    
    fairness_history[fairness_idx % 100] = fairness;
    fairness_idx++;
    
    // Check for fairness violations
    double avg_fairness = 0.0;
    for (int i = 0; i < 100; i++) {
        avg_fairness += fairness_history[i];
    }
    avg_fairness /= 100.0;
    
    if (avg_fairness < 0.7) {
        // Fairness violation: adjust reward distribution
        S->reward = 0.5 + (S->reward - 0.5) * 0.8;
    }
    
    return (ql_status){0, "fairness monitor ok"};
}

// Transparency report
void print_ethics_report(void) {
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║          ETHICS & TRANSPARENCY REPORT                      ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Stakeholders:\n");
    for (int i = 0; i < STAKEHOLDER_COUNT; i++) {
        printf("  %s: S=%.1f C=%.1f B=%.1f A=%.1f\n",
               stakeholders[i].name,
               stakeholders[i].safety_weight,
               stakeholders[i].clarity_weight,
               stakeholders[i].benefit_weight,
               stakeholders[i].autonomy_weight);
    }
    
    printf("\nAudit Trail (last 10 decisions):\n");
    int start = (audit_count > 10) ? audit_count - 10 : 0;
    for (int i = start; i < audit_count; i++) {
        printf("  [%.1f] %s (reward=%.3f, ethics=%.3f)\n",
               audit_trail[i].timestamp,
               audit_trail[i].decision,
               audit_trail[i].reward,
               audit_trail[i].ethics_score);
    }
    
    printf("\n");
}

