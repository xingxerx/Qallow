#ifndef GOVERN_H
#define GOVERN_H

#include "qallow_kernel.h"
#include "ethics.h"
#include "adaptive.h"
#include "sandbox.h"
#include <stdbool.h>

// Autonomous Governance Module
// Implements self-audit, ethics enforcement, and adaptive reinforcement loop

// Governance thresholds
#define GOVERN_ETHICS_THRESHOLD 2.9f
#define GOVERN_AUDIT_INTERVAL 100
#define GOVERN_ADAPT_INTERVAL 50
#define GOVERN_SNAPSHOT_INTERVAL 500

// Governance state tracking
typedef struct {
    float current_ethics_score;
    float previous_ethics_score;
    int audit_count;
    int violations_detected;
    bool system_stable;
    bool adaptation_active;
    double last_audit_time;
    double total_govern_time;
} govern_state_t;

// Core governance functions
void govern_init(govern_state_t* gov);
void govern_run_audit_loop(govern_state_t* gov, qallow_state_t* state, 
                           ethics_monitor_t* ethics, sandbox_manager_t* sandbox,
                           adaptive_state_t* adaptive);

// Audit and monitoring
float govern_evaluate_ethics(qallow_state_t* state, ethics_monitor_t* ethics);
bool govern_check_safety_threshold(float ethics_score);
void govern_log_audit_result(const govern_state_t* gov, float ethics_score);

// Adaptation and reinforcement
void govern_adapt_parameters(adaptive_state_t* adaptive, const govern_state_t* gov);
void govern_reinforce_learning(adaptive_state_t* adaptive, float performance_delta);

// Sandbox and isolation
bool govern_verify_sandbox_integrity(sandbox_manager_t* sandbox, qallow_state_t* state);
void govern_create_safety_checkpoint(sandbox_manager_t* sandbox, qallow_state_t* state);

// Emergency procedures
void govern_halt_on_violation(qallow_state_t* state, const char* reason);
void govern_emergency_rollback(sandbox_manager_t* sandbox, qallow_state_t* state);

// State persistence
void govern_persist_state(const govern_state_t* gov, const adaptive_state_t* adaptive);
void govern_load_state(govern_state_t* gov, adaptive_state_t* adaptive);

// Reporting
void govern_print_audit_report(const govern_state_t* gov);
void govern_print_governance_summary(const govern_state_t* gov, float final_ethics_score);

// CLI interface
int govern_cli(int argc, char** argv);

#endif // GOVERN_H

