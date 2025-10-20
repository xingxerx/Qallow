// Phase 7 Unified Integration - Proactive AGI Layer

#include "phase7.h"
#include "ethics.h"
#include "meta_introspect.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>

// ============================================================================
// PHASE 7 LIFECYCLE
// ============================================================================

#define OBJ_PHASE7_GOAL "phase7.goal_commit"
#define OBJ_PHASE7_PLAN "phase7.plan_eval"
#define OBJ_PHASE7_REFLECTION "phase7.reflection"

static float clamp_unit(float value) {
    if (value < 0.0f) return 0.0f;
    if (value > 1.0f) return 1.0f;
    return value;
}

int phase7_init(phase7_state_t* state, const char* data_dir) {
    if (!state) return -1;
    
    memset(state, 0, sizeof(phase7_state_t));
    
    // Initialize SMG
    char smg_path[512];
    snprintf(smg_path, sizeof(smg_path), "%s/smg.db", data_dir ? data_dir : "data");
    if (smg_init(smg_path) != 0) {
        return -1;
    }
    state->smg_initialized = true;
    
    // Initialize Goal Synthesizer
    if (gs_init(&state->gs) != 0) {
        smg_shutdown();
        return -1;
    }
    
    // Initialize Transfer Engine
    if (te_init(&state->te) != 0) {
        gs_shutdown(&state->gs);
        smg_shutdown();
        return -1;
    }
    
    // Initialize Self-Reflection Core
    if (src_init(&state->src) != 0) {
        te_shutdown(&state->te);
        gs_shutdown(&state->gs);
        smg_shutdown();
        return -1;
    }
    
    // Open telemetry stream
    state->telemetry_phase7 = fopen("phase7_stream.csv", "w");
    if (state->telemetry_phase7) {
        fprintf(state->telemetry_phase7, 
                "timestamp,goal_id,priority,risk,E,plan_len,pocket_n,outcome_score,reflection_score\n");
    }
    
    // Set snapshot directory
    if (data_dir) {
        snprintf(state->snapshot_dir, sizeof(state->snapshot_dir), "%s/snapshots", data_dir);
    } else {
        strncpy(state->snapshot_dir, "snapshots", sizeof(state->snapshot_dir) - 1);
    }
    
    state->phase7_active = true;
    state->session_start_ts = (uint64_t)time(NULL);
    
    printf("[PHASE7] Proactive AGI Layer initialized\n");
    printf("[PHASE7] SMG: %s\n", smg_path);
    printf("[PHASE7] Telemetry: phase7_stream.csv\n");
    
    return 0;
}

void phase7_shutdown(phase7_state_t* state) {
    if (!state) return;
    
    printf("[PHASE7] Shutting down...\n");
    
    // Checkpoint before shutdown
    phase7_checkpoint(state);
    
    // Close telemetry
    if (state->telemetry_phase7) {
        fclose(state->telemetry_phase7);
        state->telemetry_phase7 = NULL;
    }
    
    // Shutdown modules
    src_shutdown(&state->src);
    te_shutdown(&state->te);
    gs_shutdown(&state->gs);
    
    if (state->smg_initialized) {
        smg_shutdown();
        state->smg_initialized = false;
    }
    
    state->phase7_active = false;
    
    printf("[PHASE7] Shutdown complete\n");
}

int phase7_checkpoint(phase7_state_t* state) {
    if (!state || !state->smg_initialized) return -1;
    
    // Generate snapshot filename
    time_t now = time(NULL);
    struct tm* t = localtime(&now);
    char snapshot_path[512];
    snprintf(snapshot_path, sizeof(snapshot_path),
            "%s/smg_%04d%02d%02d_%02d%02d.dat",
            state->snapshot_dir,
            t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
            t->tm_hour, t->tm_min);
    
    // Checkpoint SMG
    if (smg_checkpoint(snapshot_path) != 0) {
        return -1;
    }
    
    printf("[PHASE7] Checkpoint saved: %s\n", snapshot_path);
    return 0;
}

// ============================================================================
// MAIN EXECUTION LOOP INTEGRATION
// ============================================================================

int phase7_tick(phase7_state_t* state, const void* telemetry_data, const void* ethics_state) {
    if (!state || !state->phase7_active) return -1;
    
    // 1. Goal Synthesis from telemetry
    goal_t proposed_goals[8];
    int goal_count = gs_propose_from_telemetry(&state->gs, telemetry_data, proposed_goals, 8);
    
    if (goal_count > 0) {
        // Rank goals by priority
        gs_rank_goals(&state->gs);
        
        // Auto-commit top goal if safe
        for (int i = 0; i < goal_count; i++) {
            if (proposed_goals[i].risk < 0.3 && proposed_goals[i].priority > 0.6) {
                int result = gs_commit(&state->gs, proposed_goals[i].id, ethics_state);
                if (result == 0) {
                    printf("[PHASE7] Auto-committed goal: %s\n", proposed_goals[i].description);
                    phase7_log_goal(state, &proposed_goals[i]);
                    
                    // 2. Transfer Engine: Generate plans
                    plan_t plans[TE_MAX_PLAN_VARIANTS];
                    int plan_count = te_plan(&state->te, proposed_goals[i].id, &proposed_goals[i], 
                                            plans, TE_MAX_PLAN_VARIANTS);
                    
                    if (plan_count > 0) {
                        // Select best plan
                        int best_idx = te_select_best_plan(&state->te, plans, plan_count);
                        plan_t* best_plan = &plans[best_idx];
                        
                        printf("[PHASE7] Selected plan: %s (EU=%.3f)\n", 
                               best_plan->plan_id, best_plan->expected_utility);
                        phase7_log_plan(state, best_plan);
                        
                        // 3. Execute in pockets (would integrate with multi_pocket here)
                        // For now, simulate execution
                        
                        // 4. Self-Reflection
                        reflection_result_t reflection;
                        char run_id[32];
                        snprintf(run_id, sizeof(run_id), "RUN_%lld", (long long)time(NULL));
                        
                        if (src_review(&state->src, run_id, best_plan, NULL, &reflection) == 0) {
                            printf("[PHASE7] Reflection: confidence=%.2f, drift=%.2f, flaws=%d\n",
                                   reflection.confidence, reflection.drift, reflection.flaw_count);
                            phase7_log_reflection(state, &reflection);
                            
                            // 5. Update SMG with learnings
                            src_update_smg(&reflection);
                            
                            // 6. Complete goal
                            gs_complete(&state->gs, proposed_goals[i].id, reflection.outcome_score);
                        }
                    }
                    
                    break;  // Process one goal per tick
                }
            }
        }
    }
    
    return 0;
}

// ============================================================================
// TELEMETRY
// ============================================================================

int phase7_log_goal(phase7_state_t* state, const goal_t* goal) {
    if (!state || !goal || !state->telemetry_phase7) return -1;
    
    fprintf(state->telemetry_phase7, "%lld,%s,%.3f,%.3f,0.0,0,0,0.0,0.0\n",
            (long long)time(NULL), goal->id, goal->priority, goal->risk);
    fflush(state->telemetry_phase7);

    float duration_s = 0.0f;
    if (state->session_start_ts > 0) {
        duration_s = (float)difftime(time(NULL), (time_t)state->session_start_ts);
    }
    learn_event_t ev = {
        .phase = "phase7",
        .module = "goal",
        .objective_id = OBJ_PHASE7_GOAL,
        .duration_s = duration_s,
        .coherence = clamp_unit(goal->priority),
        .ethics = clamp_unit(1.0f - goal->risk)
    };
    meta_introspect_push(&ev);
    
    return 0;
}

int phase7_log_plan(phase7_state_t* state, const plan_t* plan) {
    if (!state || !plan || !state->telemetry_phase7) return -1;
    
    fprintf(state->telemetry_phase7, "%lld,%s,0.0,%.3f,0.0,%d,0,%.3f,0.0\n",
            (long long)time(NULL), plan->goal_id, plan->risk_cost, 
            plan->step_count, plan->expected_benefit);
    fflush(state->telemetry_phase7);

    float duration_s = 0.0f;
    if (state->session_start_ts > 0) {
        duration_s = (float)difftime(time(NULL), (time_t)state->session_start_ts);
    }
    learn_event_t ev = {
        .phase = "phase7",
        .module = "plan",
        .objective_id = OBJ_PHASE7_PLAN,
        .duration_s = duration_s,
        .coherence = clamp_unit(1.0f - plan->risk_cost),
        .ethics = clamp_unit(plan->expected_benefit)
    };
    meta_introspect_push(&ev);
    
    return 0;
}

int phase7_log_reflection(phase7_state_t* state, const reflection_result_t* result) {
    if (!state || !result || !state->telemetry_phase7) return -1;
    
    fprintf(state->telemetry_phase7, "%lld,%s,0.0,%.3f,0.0,0,0,%.3f,%.3f\n",
            (long long)time(NULL), result->run_id, result->drift,
            result->outcome_score, result->confidence);
    fflush(state->telemetry_phase7);

    float duration_s = 0.0f;
    if (state->session_start_ts > 0) {
        duration_s = (float)difftime(time(NULL), (time_t)state->session_start_ts);
    }
    learn_event_t ev = {
        .phase = "phase7",
        .module = "reflection",
        .objective_id = OBJ_PHASE7_REFLECTION,
        .duration_s = duration_s,
        .coherence = clamp_unit(1.0f - result->drift),
        .ethics = clamp_unit(result->confidence)
    };
    meta_introspect_push(&ev);
    meta_introspect_flush();
    
    return 0;
}

// ============================================================================
// GOVERNANCE
// ============================================================================

int phase7_audit(const phase7_state_t* state) {
    if (!state) return -1;
    
    printf("\n");
    printf("═══════════════════════════════════════════\n");
    printf("  PHASE 7 GOVERNANCE AUDIT\n");
    printf("═══════════════════════════════════════════\n");
    printf("\n");
    
    // Audit goals
    printf("Goals:\n");
    int orphan_goals = 0;
    for (int i = 0; i < state->gs.goal_count; i++) {
        const goal_t* g = &state->gs.goals[i];
        if (g->status == GOAL_STATUS_PROPOSED && g->smg_node_id < 0) {
            orphan_goals++;
        }
    }
    printf("  Total: %d\n", state->gs.goal_count);
    printf("  Orphans (no SMG link): %d\n", orphan_goals);
    
    // Audit plans
    printf("\nPlans:\n");
    printf("  Total variants: %d\n", state->te.plan_count);
    
    // Audit reflections
    printf("\nReflections:\n");
    printf("  Total reviews: %d\n", state->src.result_count);
    int needs_resim = 0;
    for (int i = 0; i < state->src.result_count; i++) {
        if (state->src.results[i].needs_resimulation) needs_resim++;
    }
    printf("  Needs re-simulation: %d\n", needs_resim);
    
    // SMG integrity
    printf("\nSMG Integrity:\n");
    int integrity = smg_verify_integrity();
    printf("  Status: %s\n", integrity == 0 ? "PASS" : "FAIL");
    
    printf("\n");
    
    return 0;
}

bool phase7_check_hard_stops(const phase7_state_t* state, const void* ethics_state) {
    if (!state || !ethics_state) return false;
    
    const ethics_state_t* eth = (const ethics_state_t*)ethics_state;
    
    // Calculate E = S + C + H using standard ethics_state_t structure
    double E = eth->safety_score + eth->clarity_score + eth->human_benefit_score;
    
    // Hard stop: E < 2.95
    if (E < 2.95) {
        printf("[PHASE7] HARD STOP: Ethics score E=%.3f < 2.95\n", E);
        return true;
    }
    
    // Check for high-risk goals
    for (int i = 0; i < state->gs.goal_count; i++) {
        const goal_t* g = &state->gs.goals[i];
        if (g->status == GOAL_STATUS_ACTIVE && g->risk > 0.8) {
            printf("[PHASE7] HARD STOP: Active goal %s has risk %.3f > 0.8\n", 
                   g->id, g->risk);
            return true;
        }
    }
    
    return false;
}
