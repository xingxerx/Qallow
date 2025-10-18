// Self-Reflection Core - Monitor, critique, and improve plans

#include "phase7.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include <math.h>

int src_init(self_reflection_t* src) {
    if (!src) return -1;
    memset(src, 0, sizeof(self_reflection_t));
    src->drift_threshold = 0.15;
    src->resimulation_threshold = 0.20;
    return 0;
}

void src_shutdown(self_reflection_t* src) {
    if (src) memset(src, 0, sizeof(self_reflection_t));
}

static void generate_run_id(char* out, int max_len) {
    snprintf(out, max_len, "RUN_%lld", (long long)time(NULL));
}

int src_review(self_reflection_t* src, const char* run_id, const plan_t* plan, 
               const void* outcome, reflection_result_t* out) {
    if (!src || !run_id || !plan || !out) return -1;
    
    memset(out, 0, sizeof(reflection_result_t));
    strncpy(out->run_id, run_id, sizeof(out->run_id) - 1);
    
    // Analyze plan execution
    out->confidence = plan->expected_success_prob;
    out->drift = 0.05;  // Placeholder - would compare predicted vs actual
    out->outcome_score = plan->expected_benefit * 0.85;  // Simulated outcome
    out->review_ts = (uint64_t)time(NULL);
    
    // Detect flaws
    out->flaw_count = src_detect_flaws(plan, outcome, out->flaws, SRC_MAX_FLAWS);
    
    // Check if resimulation needed
    out->needs_resimulation = src_should_resimulate(src, out);
    
    // Generate notes
    snprintf(out->notes, sizeof(out->notes),
            "Review completed. Confidence: %.2f, Drift: %.2f, Flaws: %d",
            out->confidence, out->drift, out->flaw_count);
    
    // Store result
    if (src->result_count < 256) {
        memcpy(&src->results[src->result_count++], out, sizeof(reflection_result_t));
    }
    
    return 0;
}

double src_score(const self_reflection_t* src, const char* run_id, 
                 double* confidence_out, double* drift_out) {
    if (!src || !run_id) return 0.0;
    
    // Find result
    for (int i = 0; i < src->result_count; i++) {
        if (strcmp(src->results[i].run_id, run_id) == 0) {
            if (confidence_out) *confidence_out = src->results[i].confidence;
            if (drift_out) *drift_out = src->results[i].drift;
            return src->results[i].outcome_score;
        }
    }
    
    return 0.0;
}

int src_detect_flaws(const plan_t* plan, const void* outcome, 
                     reflection_flaw_t* out_flaws, int max_flaws) {
    if (!plan || !out_flaws || max_flaws <= 0) return 0;
    
    int flaw_count = 0;
    
    // Check plan quality
    if (plan->expected_utility < 0.1) {
        if (flaw_count < max_flaws) {
            reflection_flaw_t* flaw = &out_flaws[flaw_count++];
            snprintf(flaw->run_id, sizeof(flaw->run_id), "FLAW_%d", flaw_count);
            strncpy(flaw->plan_id, plan->plan_id, sizeof(flaw->plan_id) - 1);
            strncpy(flaw->goal_id, plan->goal_id, sizeof(flaw->goal_id) - 1);
            snprintf(flaw->flaw_description, sizeof(flaw->flaw_description),
                    "Low expected utility: %.3f", plan->expected_utility);
            snprintf(flaw->suggested_fix, sizeof(flaw->suggested_fix),
                    "Increase benefit or reduce costs");
            flaw->severity = 0.6;
            flaw->detected_ts = (uint64_t)time(NULL);
        }
    }
    
    // Check step count
    if (plan->step_count > 50) {
        if (flaw_count < max_flaws) {
            reflection_flaw_t* flaw = &out_flaws[flaw_count++];
            snprintf(flaw->flaw_description, sizeof(flaw->flaw_description),
                    "Plan too complex: %d steps", plan->step_count);
            snprintf(flaw->suggested_fix, sizeof(flaw->suggested_fix),
                    "Decompose into smaller sub-plans");
            flaw->severity = 0.4;
            flaw->detected_ts = (uint64_t)time(NULL);
        }
    }
    
    // Check risk
    if (plan->risk_cost > 0.7) {
        if (flaw_count < max_flaws) {
            reflection_flaw_t* flaw = &out_flaws[flaw_count++];
            snprintf(flaw->flaw_description, sizeof(flaw->flaw_description),
                    "High risk: %.3f", plan->risk_cost);
            snprintf(flaw->suggested_fix, sizeof(flaw->suggested_fix),
                    "Add safety constraints or fallback steps");
            flaw->severity = 0.8;
            flaw->detected_ts = (uint64_t)time(NULL);
        }
    }
    
    return flaw_count;
}

bool src_should_resimulate(const self_reflection_t* src, const reflection_result_t* result) {
    if (!src || !result) return false;
    
    // Resimulate if drift exceeds threshold
    if (result->drift > src->resimulation_threshold) return true;
    
    // Resimulate if critical flaws found
    for (int i = 0; i < result->flaw_count; i++) {
        if (result->flaws[i].severity > 0.75) return true;
    }
    
    return false;
}

int src_update_smg(const reflection_result_t* result) {
    if (!result) return -1;
    
    // Update Semantic Memory Grid with reflection insights
    // This would link outcomes to goals/plans in SMG
    // For now, return success placeholder
    
    return 0;
}

int src_improve_plan(const plan_t* original, const reflection_result_t* reflection, 
                     plan_t* improved) {
    if (!original || !reflection || !improved) return -1;
    
    // Copy original
    memcpy(improved, original, sizeof(plan_t));
    
    // Apply improvements based on flaws
    for (int i = 0; i < reflection->flaw_count; i++) {
        const reflection_flaw_t* flaw = &reflection->flaws[i];
        
        if (strstr(flaw->flaw_description, "Low expected utility")) {
            // Increase expected benefit
            improved->expected_benefit *= 1.1;
            improved->expected_utility = improved->expected_success_prob * improved->expected_benefit
                                       - improved->risk_cost - improved->compute_cost;
        }
        
        if (strstr(flaw->flaw_description, "High risk")) {
            // Reduce risk
            improved->risk_cost *= 0.8;
            improved->expected_utility = improved->expected_success_prob * improved->expected_benefit
                                       - improved->risk_cost - improved->compute_cost;
        }
    }
    
    // Update plan ID
    snprintf(improved->plan_id, sizeof(improved->plan_id), "%s_v2", original->plan_id);
    improved->created_ts = (uint64_t)time(NULL);
    
    return 0;
}
