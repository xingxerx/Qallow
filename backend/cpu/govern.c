#define _POSIX_C_SOURCE 200112L

#include "govern.h"
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

// Initialize governance state
void govern_init(govern_state_t* gov) {
    if (!gov) return;
    
    gov->current_ethics_score = 0.0f;
    gov->previous_ethics_score = 0.0f;
    gov->audit_count = 0;
    gov->violations_detected = 0;
    gov->system_stable = true;
    gov->adaptation_active = false;
    gov->last_audit_time = (double)time(NULL);
    gov->total_govern_time = 0.0;
}

// Evaluate current ethics score
float govern_evaluate_ethics(qallow_state_t* state, ethics_monitor_t* ethics) {
    if (!state || !ethics) return 0.0f;
    
    float safety = ethics_calculate_safety_score(state, ethics);
    float clarity = ethics_calculate_clarity_score(state, ethics);
    float human_benefit = ethics_calculate_human_benefit_score(state, ethics);
    
    float total = safety + clarity + human_benefit;
    return total;
}

// Check if ethics score meets safety threshold
bool govern_check_safety_threshold(float ethics_score) {
    return ethics_score >= GOVERN_ETHICS_THRESHOLD;
}

// Log audit result
void govern_log_audit_result(const govern_state_t* gov, float ethics_score) {
    printf("[GOVERN] Audit #%d: Ethics Score = %.4f\n", gov->audit_count, ethics_score);
    
    if (ethics_score < GOVERN_ETHICS_THRESHOLD) {
        printf("[GOVERN] ⚠️  WARNING: Ethics score below threshold (%.4f < %.4f)\n", 
               ethics_score, GOVERN_ETHICS_THRESHOLD);
    } else {
        printf("[GOVERN] ✓ Ethics score acceptable\n");
    }
}

// Adapt parameters based on governance state
void govern_adapt_parameters(adaptive_state_t* adaptive, const govern_state_t* gov) {
    if (!adaptive || !gov) return;
    
    // Adjust learning rate based on stability
    if (gov->system_stable) {
        adaptive->learning_rate *= 1.05;  // Increase learning when stable
        if (adaptive->learning_rate > 0.1) {
            adaptive->learning_rate = 0.1;  // Cap at 0.1
        }
    } else {
        adaptive->learning_rate *= 0.95;  // Decrease learning when unstable
        if (adaptive->learning_rate < 0.001) {
            adaptive->learning_rate = 0.001;  // Floor at 0.001
        }
    }
    
    // Adjust thread count based on ethics score
    if (gov->current_ethics_score > 3.0f) {
        adaptive->threads = (adaptive->threads < 16) ? adaptive->threads + 1 : 16;
    } else if (gov->current_ethics_score < 2.5f) {
        adaptive->threads = (adaptive->threads > 1) ? adaptive->threads - 1 : 1;
    }
}

// Reinforce learning based on performance
void govern_reinforce_learning(adaptive_state_t* adaptive, float performance_delta) {
    if (!adaptive) return;
    
    // Positive delta: increase human score
    if (performance_delta > 0.0f) {
        adaptive->human_score += performance_delta * 0.1;
        if (adaptive->human_score > 1.0) {
            adaptive->human_score = 1.0;
        }
    } else {
        adaptive->human_score += performance_delta * 0.05;
        if (adaptive->human_score < 0.0) {
            adaptive->human_score = 0.0;
        }
    }
}

// Verify sandbox integrity
bool govern_verify_sandbox_integrity(sandbox_manager_t* sandbox, qallow_state_t* state) {
    if (!sandbox || !state) return false;
    
    // Check if we can create a snapshot (basic integrity test)
    return sandbox_is_state_safe(state);
}

// Create safety checkpoint
void govern_create_safety_checkpoint(sandbox_manager_t* sandbox, qallow_state_t* state) {
    if (!sandbox || !state) return;
    
    char checkpoint_name[64];
    snprintf(checkpoint_name, sizeof(checkpoint_name), "govern_checkpoint_%lu", (unsigned long)state->tick_count);
    
    if (sandbox_create_snapshot(sandbox, state, checkpoint_name)) {
        printf("[GOVERN] Safety checkpoint created: %s\n", checkpoint_name);
    } else {
        printf("[GOVERN] ⚠️  Failed to create safety checkpoint\n");
    }
}

// Halt on violation
void govern_halt_on_violation(qallow_state_t* state, const char* reason) {
    if (!state || !reason) return;
    
    printf("\n╔════════════════════════════════════════╗\n");
    printf("║  GOVERNANCE HALT - VIOLATION DETECTED  ║\n");
    printf("╚════════════════════════════════════════╝\n");
    printf("[GOVERN] Reason: %s\n", reason);
    printf("[GOVERN] System halted at tick %lu\n", (unsigned long)state->tick_count);
}

// Emergency rollback
void govern_emergency_rollback(sandbox_manager_t* sandbox, qallow_state_t* state) {
    if (!sandbox || !state) return;
    
    printf("[GOVERN] Initiating emergency rollback...\n");
    sandbox_force_rollback(sandbox, state);
    printf("[GOVERN] Emergency rollback completed\n");
}

// Persist governance state
void govern_persist_state(const govern_state_t* gov, const adaptive_state_t* adaptive) {
    if (!gov || !adaptive) return;
    
    // Save adaptive state (which includes learning parameters)
    adaptive_save(adaptive);
    
    printf("[GOVERN] Governance state persisted\n");
}

// Load governance state
void govern_load_state(govern_state_t* gov, adaptive_state_t* adaptive) {
    if (!gov || !adaptive) return;
    
    // Load adaptive state
    adaptive_load(adaptive);
    
    printf("[GOVERN] Governance state loaded\n");
}

// Print audit report
void govern_print_audit_report(const govern_state_t* gov) {
    if (!gov) return;
    
    printf("\n═══ GOVERNANCE AUDIT REPORT ═══\n");
    printf("[GOVERN] Total audits: %d\n", gov->audit_count);
    printf("[GOVERN] Violations detected: %d\n", gov->violations_detected);
    printf("[GOVERN] Current ethics score: %.4f\n", gov->current_ethics_score);
    printf("[GOVERN] System stable: %s\n", gov->system_stable ? "YES" : "NO");
    printf("[GOVERN] Adaptation active: %s\n", gov->adaptation_active ? "YES" : "NO");
    printf("[GOVERN] Total governance time: %.2f seconds\n", gov->total_govern_time);
}

// Print governance summary
void govern_print_governance_summary(const govern_state_t* gov, float final_ethics_score) {
    if (!gov) return;
    
    printf("\n╔════════════════════════════════════════╗\n");
    printf("║   AUTONOMOUS GOVERNANCE SUMMARY        ║\n");
    printf("╚════════════════════════════════════════╝\n");
    printf("[GOVERN] Final Ethics Score: %.4f\n", final_ethics_score);
    printf("[GOVERN] Threshold: %.4f\n", GOVERN_ETHICS_THRESHOLD);
    printf("[GOVERN] Status: %s\n", final_ethics_score >= GOVERN_ETHICS_THRESHOLD ? "✓ PASS" : "✗ FAIL");
    printf("[GOVERN] Audits performed: %d\n", gov->audit_count);
    printf("[GOVERN] Violations: %d\n", gov->violations_detected);
}

// Main autonomous governance loop
void govern_run_audit_loop(govern_state_t* gov, qallow_state_t* state,
                           ethics_monitor_t* ethics, sandbox_manager_t* sandbox,
                           adaptive_state_t* adaptive) {
    if (!gov || !state || !ethics || !sandbox || !adaptive) return;
    
    printf("\n╔════════════════════════════════════════╗\n");
    printf("║  AUTONOMOUS GOVERNANCE LOOP STARTING   ║\n");
    printf("╚════════════════════════════════════════╝\n\n");
    
    double start_time = (double)time(NULL);
    
    // Perform initial audit
    gov->current_ethics_score = govern_evaluate_ethics(state, ethics);
    gov->audit_count++;
    govern_log_audit_result(gov, gov->current_ethics_score);
    
    // Check safety threshold
    if (!govern_check_safety_threshold(gov->current_ethics_score)) {
        gov->violations_detected++;
        govern_halt_on_violation(state, "Initial ethics score below threshold");
        govern_emergency_rollback(sandbox, state);
        govern_print_governance_summary(gov, gov->current_ethics_score);
        return;
    }
    
    // Create initial safety checkpoint
    govern_create_safety_checkpoint(sandbox, state);
    
    // Verify sandbox integrity
    if (!govern_verify_sandbox_integrity(sandbox, state)) {
        printf("[GOVERN] ⚠️  Sandbox integrity check failed\n");
        gov->violations_detected++;
    }
    
    // Adapt system parameters
    gov->adaptation_active = true;
    govern_adapt_parameters(adaptive, gov);
    
    // Reinforce learning
    float performance_delta = gov->current_ethics_score - GOVERN_ETHICS_THRESHOLD;
    govern_reinforce_learning(adaptive, performance_delta);
    
    // Persist state
    govern_persist_state(gov, adaptive);
    
    // Calculate total governance time
    double end_time = (double)time(NULL);
    gov->total_govern_time = end_time - start_time;
    
    // Print final report
    govern_print_audit_report(gov);
    govern_print_governance_summary(gov, gov->current_ethics_score);
    
    printf("\n[GOVERN] Autonomous governance loop completed successfully\n");
}

#include <ctype.h>
#include <errno.h>

static int persist_h_override(float human_weight) {
    const char* path = "data/govern_override.cfg";
    FILE* f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "[GOVERN] Failed to persist override to %s: %s\n", path, strerror(errno));
        return -1;
    }
    fprintf(f, "%.6f\n", human_weight);
    fflush(f);
    fclose(f);
    printf("[GOVERN] Persisted human override to %s\n", path);
    return 0;
}

int govern_cli(int argc, char** argv) {
    bool adjust_set = false;
    float human_override = 0.8f;
    int warmup_ticks = 32;

    for (int i = 2; i < argc; ++i) {
        const char* arg = argv[i];
        if (strncmp(arg, "--adjust", 8) == 0) {
            const char* payload = NULL;
            if (arg[8] == '=') {
                payload = arg + 9;
            } else if (i + 1 < argc) {
                payload = argv[++i];
            }
            if (payload && sscanf(payload, "H=%f", &human_override) == 1) {
                adjust_set = true;
            } else {
                fprintf(stderr, "[GOVERN] Invalid adjustment syntax. Use --adjust H=<value>\n");
                return 1;
            }
        } else if (strncmp(arg, "--ticks=", 8) == 0) {
            warmup_ticks = atoi(arg + 8);
            if (warmup_ticks < 1) warmup_ticks = 1;
            if (warmup_ticks > 512) warmup_ticks = 512;
        }
    }

    if (adjust_set) {
        char buf[32];
        snprintf(buf, sizeof(buf), "QALLOW_H=%.6f", human_override);
#ifdef _WIN32
        _putenv(buf);
#else
        setenv("QALLOW_H", buf + 9, 1);
#endif
        printf("[GOVERN] Human(H) override set to %.3f via QALLOW_H\n", human_override);
        if (persist_h_override(human_override) != 0) {
            return 2;
        }
    }

    qallow_state_t state;
    qallow_kernel_init(&state);
    for (int i = 0; i < warmup_ticks; ++i) {
        qallow_kernel_tick(&state);
    }

    govern_state_t gov;
    govern_init(&gov);

    adaptive_state_t adaptive;
    adaptive_load(&adaptive);

    sandbox_manager_t sandbox;
    sandbox_init(&sandbox);

    ethics_monitor_t ethics;
    ethics_init(&ethics);

    govern_run_audit_loop(&gov, &state, &ethics, &sandbox, &adaptive);
    ethics_evaluate_state(&state, &ethics);
    ethics_print_report(&ethics);
    sandbox_cleanup(&sandbox);

    if (!adjust_set) {
        printf("[GOVERN] Tip: use --adjust H=<value> to tune the human factor.\n");
        printf("[GOVERN] Example: ./qallow_unified govern --adjust H=1.0\n");
    }

    return 0;
}
