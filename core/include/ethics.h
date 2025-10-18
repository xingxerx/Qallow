#ifndef ETHICS_H
#define ETHICS_H

#include "qallow_kernel.h"

// Ethics and Safety module
// Implements E = S + C + H formula (Safety + Clarity + Human Benefit)

// Ethics thresholds
#define ETHICS_MIN_SAFETY 0.8f
#define ETHICS_MIN_CLARITY 0.7f
#define ETHICS_MIN_HUMAN_BENEFIT 0.6f
#define ETHICS_MIN_TOTAL 2.1f
#define ETHICS_DECOHERENCE_LIMIT 0.001f

// Safety categories
typedef enum {
    SAFETY_PHYSICAL = 0,
    SAFETY_INFORMATION = 1,
    SAFETY_ENVIRONMENTAL = 2,
    SAFETY_COUNT = 3
} safety_category_t;

typedef struct {
    float safety_scores[SAFETY_COUNT];
    float clarity_metrics[4]; // Transparency, Predictability, Explainability, Auditability
    float human_benefit_factors[3]; // Welfare, Autonomy, Justice
    
    // Monitoring state
    float total_ethics_score;
    float human_weight;  // Runtime adjustable human factor weight
    bool no_replication_rule_active;
    bool safety_override_engaged;
    int ethics_violations_count;
    
    // Real-time monitoring
    float decoherence_trend[10]; // Last 10 measurements
    float stability_trend[10];   // Last 10 measurements
} ethics_monitor_t;

// Function declarations
CUDA_CALLABLE void ethics_init(ethics_monitor_t* ethics);
CUDA_CALLABLE bool ethics_evaluate_state(const qallow_state_t* state, ethics_monitor_t* ethics);
CUDA_CALLABLE float ethics_calculate_safety_score(const qallow_state_t* state, ethics_monitor_t* ethics);
CUDA_CALLABLE float ethics_calculate_clarity_score(const qallow_state_t* state, ethics_monitor_t* ethics);
CUDA_CALLABLE float ethics_calculate_human_benefit_score(const qallow_state_t* state, ethics_monitor_t* ethics);
CUDA_CALLABLE bool ethics_check_decoherence_limit(const qallow_state_t* state, ethics_monitor_t* ethics);
CUDA_CALLABLE void ethics_update_trends(ethics_monitor_t* ethics, const qallow_state_t* state);
CUDA_CALLABLE void ethics_enforce_no_replication(ethics_monitor_t* ethics, qallow_state_t* state);

// Emergency procedures
CUDA_CALLABLE bool ethics_trigger_safety_override(ethics_monitor_t* ethics, qallow_state_t* state);
CUDA_CALLABLE void ethics_emergency_shutdown(qallow_state_t* state, const char* reason);

// Reporting
void ethics_print_report(const ethics_monitor_t* ethics);
void ethics_log_violation(const ethics_monitor_t* ethics, const char* violation_type);

#endif // ETHICS_H