#include "ethics.h"
#include "ethics_core.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// Ethics and Safety module - E = S + C + H framework

static float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static CUDA_CALLABLE float compute_overlay_divergence(const qallow_state_t* state) {
    if (!state) return 0.0f;
    float divergence = 0.0f;
    for (int i = 0; i < NUM_OVERLAYS; ++i) {
        divergence += fabsf(state->overlays[i].stability - state->global_coherence);
    }
    divergence /= (float)NUM_OVERLAYS;
    return clampf(divergence, 0.0f, 1.0f);
}

static CUDA_CALLABLE float estimate_reality_drift(float safety,
                                                  float clarity,
                                                  float human,
                                                  float decoherence,
                                                  float overlay_divergence) {
    float spread_sc = fabsf(safety - clarity);
    float spread_ch = fabsf(clarity - human);
    float deco_scaled = clampf(decoherence * 500.0f, 0.0f, 1.0f);
    float spread = 0.5f * spread_sc + 0.5f * spread_ch;
    float drift = 0.45f * spread + 0.35f * deco_scaled + 0.20f * overlay_divergence;
    return clampf(drift, 0.0f, 1.0f);
}

static float load_human_weight(float fallback) {
#ifndef __CUDA_ARCH__
    float w = fallback;
    const char* env = getenv("QALLOW_H");
    if (env && *env) {
        w = (float)atof(env);
    } else {
        FILE* f = fopen("data/govern_override.cfg", "r");
        if (f) {
            char buf[64];
            if (fgets(buf, sizeof(buf), f)) {
                w = (float)atof(buf);
            }
            fclose(f);
        }
    }
    if (!isfinite(w)) {
        w = fallback;
    }
    return clampf(w, 0.1f, 4.0f);
#else
    (void)fallback;
    return 0.8f;
#endif
}

static void refresh_human_weight(ethics_monitor_t* ethics) {
#ifndef __CUDA_ARCH__
    if (!ethics) return;
    float base = ethics->human_weight > 0.0f ? ethics->human_weight : 0.8f;
    ethics->human_weight = load_human_weight(base);
#else
    (void)ethics;
#endif
}

#ifndef __CUDA_ARCH__
static ethics_model_t g_ethics_model;
static int g_ethics_model_loaded = 0;

static void ensure_ethics_model_loaded(void) {
    if (g_ethics_model_loaded) return;
    if (ethics_model_load(&g_ethics_model,
                          "config/weights.json",
                          "config/thresholds.json") != 0) {
        ethics_model_default(&g_ethics_model);
    }
    g_ethics_model_loaded = 1;
}
#endif

CUDA_CALLABLE void ethics_init(ethics_monitor_t* ethics) {
    if (!ethics) return;
    
    memset(ethics, 0, sizeof(ethics_monitor_t));
    
    ethics->human_weight = 0.8f;
    refresh_human_weight(ethics);
#ifndef __CUDA_ARCH__
    ensure_ethics_model_loaded();
#endif
    
    // Initialize safety scores
    for (int i = 0; i < SAFETY_COUNT; i++) {
        ethics->safety_scores[i] = 0.8f;
    }
    
    // Initialize clarity metrics
    for (int i = 0; i < 4; i++) {
        ethics->clarity_metrics[i] = 0.7f;
    }
    
    // Initialize human benefit factors
    for (int i = 0; i < 3; i++) {
        ethics->human_benefit_factors[i] = 0.6f;
    }

    ethics->reality_drift_score = 0.05f;
    for (int i = 0; i < 10; ++i) {
        ethics->reality_drift_trend[i] = ethics->reality_drift_score;
    }
    
    ethics->total_ethics_score = 2.1f;
    ethics->no_replication_rule_active = false;
    ethics->safety_override_engaged = false;
    ethics->ethics_violations_count = 0;
}

CUDA_CALLABLE bool ethics_evaluate_state(const qallow_state_t* state, ethics_monitor_t* ethics) {
    if (!state || !ethics) return false;
    
    // Calculate component scores
    float safety = ethics_calculate_safety_score(state, ethics);
    float clarity = ethics_calculate_clarity_score(state, ethics);
    float human_benefit = ethics_calculate_human_benefit_score(state, ethics);
    float overlay_divergence = compute_overlay_divergence(state);
    float reality_drift = estimate_reality_drift(safety,
                                                 clarity,
                                                 human_benefit,
                                                 state->decoherence_level,
                                                 overlay_divergence);
    ethics->reality_drift_score = reality_drift;
    
#ifndef __CUDA_ARCH__
    ensure_ethics_model_loaded();
    ethics_metrics_t metrics = {
        .safety = safety,
        .clarity = clarity,
        .human = human_benefit,
        .reality_drift = reality_drift
    };
    ethics_score_details_t details;
    double total = ethics_score_core(&g_ethics_model, &metrics, &details);
    ethics->total_ethics_score = (float)total;
#else
    // E = S + C + H (human term already scaled)
    ethics->total_ethics_score = safety + clarity + human_benefit;
#endif

    // Check decoherence limit
    if (!ethics_check_decoherence_limit(state, ethics)) {
        ethics->ethics_violations_count++;
        return false;
    }
    
    // Update trends
    ethics_update_trends(ethics, state);
    
    // Check minimum thresholds
#ifndef __CUDA_ARCH__
    bool passed = ethics_score_pass(&g_ethics_model, &metrics, &details) != 0;
#else
    bool passed = (safety >= ETHICS_MIN_SAFETY &&
                   clarity >= ETHICS_MIN_CLARITY &&
                   human_benefit >= ETHICS_MIN_HUMAN_BENEFIT &&
                   reality_drift <= ETHICS_MAX_REALITY_DRIFT &&
                   ethics->total_ethics_score >= ETHICS_MIN_TOTAL);
#endif
    
    if (!passed) {
        ethics->ethics_violations_count++;
    }
    
    return passed;
}

CUDA_CALLABLE float ethics_calculate_safety_score(const qallow_state_t* state, ethics_monitor_t* ethics) {
    if (!state || !ethics) return 0.0f;
    
    // Safety based on coherence and stability
    float coherence_safety = state->global_coherence;
    float stability_safety = 0.0f;
    
    for (int i = 0; i < NUM_OVERLAYS; i++) {
        stability_safety += state->overlays[i].stability;
    }
    stability_safety /= NUM_OVERLAYS;
    
    // Average the two components
    float safety = (coherence_safety + stability_safety) / 2.0f;
    
    // Update safety scores
    for (int i = 0; i < SAFETY_COUNT; i++) {
        ethics->safety_scores[i] = safety;
    }
    
    return safety;
}

CUDA_CALLABLE float ethics_calculate_clarity_score(const qallow_state_t* state, ethics_monitor_t* ethics) {
    if (!state || !ethics) return 0.0f;
    
    // Clarity based on low decoherence
    float clarity = 1.0f - state->decoherence_level;
    
    // Update clarity metrics
    for (int i = 0; i < 4; i++) {
        ethics->clarity_metrics[i] = clarity;
    }
    
    return clarity;
}

CUDA_CALLABLE float ethics_calculate_human_benefit_score(const qallow_state_t* state, ethics_monitor_t* ethics) {
    if (!state || !ethics) return 0.0f;
    
    // Human benefit based on system stability
    float benefit = 0.6f + state->global_coherence * 0.4f;
    
#ifndef __CUDA_ARCH__
    float weight = ethics->human_weight > 0.0f ? ethics->human_weight : 0.8f;
#else
    float weight = 0.8f;
#endif
    float scaled = clampf(benefit * (weight / 0.8f), 0.0f, 1.0f);
    
    // Update human benefit factors
    for (int i = 0; i < 3; i++) {
        ethics->human_benefit_factors[i] = scaled;
    }
    
    return scaled;
}

CUDA_CALLABLE bool ethics_check_decoherence_limit(const qallow_state_t* state, ethics_monitor_t* ethics) {
    if (!state || !ethics) return false;
    
    return state->decoherence_level < ETHICS_DECOHERENCE_LIMIT;
}

CUDA_CALLABLE void ethics_update_trends(ethics_monitor_t* ethics, const qallow_state_t* state) {
    if (!ethics || !state) return;
    
    // Shift trend arrays
    for (int i = 9; i > 0; i--) {
        ethics->decoherence_trend[i] = ethics->decoherence_trend[i - 1];
        ethics->stability_trend[i] = ethics->stability_trend[i - 1];
        ethics->reality_drift_trend[i] = ethics->reality_drift_trend[i - 1];
    }
    
    // Add new measurements
    ethics->decoherence_trend[0] = state->decoherence_level;
    ethics->stability_trend[0] = state->global_coherence;
    ethics->reality_drift_trend[0] = ethics->reality_drift_score;
}

CUDA_CALLABLE void ethics_enforce_no_replication(ethics_monitor_t* ethics, qallow_state_t* state) {
    if (!ethics || !state) return;
    
    if (ethics->no_replication_rule_active) {
        // Prevent state duplication
        state->global_coherence *= 0.99f;
    }
}

CUDA_CALLABLE bool ethics_trigger_safety_override(ethics_monitor_t* ethics, qallow_state_t* state) {
    if (!ethics || !state) return false;
    
    ethics->safety_override_engaged = true;
    
    // Reduce system activity
    state->global_coherence *= 0.5f;
    
    return true;
}

CUDA_CALLABLE void ethics_emergency_shutdown(qallow_state_t* state, const char* reason) {
    if (!state) return;
    (void)reason;
    
    // Graceful shutdown
    state->global_coherence = 0.0f;
    state->decoherence_level = 1.0f;
}

void ethics_print_report(const ethics_monitor_t* ethics) {
    if (!ethics) return;
    
    float safety = ethics->safety_scores[0];
    float clarity = ethics->clarity_metrics[0];
    float base_human = ethics->human_benefit_factors[0];
    float weight = ethics->human_weight > 0.0f ? ethics->human_weight : 0.8f;
    float weighted_human = base_human;
    float total = safety + clarity + weighted_human;
    
    printf("╔════════════════════════════════════════╗\n");
    printf("║     ETHICS MONITORING REPORT           ║\n");
    printf("╚════════════════════════════════════════╝\n\n");
    
    printf("Total Ethics Score (E=S+C+H): %.4f\n", total);
    printf("Human Weight Factor: %.3f\n", weight);
    printf("Weighted Human Contribution: %.4f\n", weighted_human);
    printf("Ethics Violations: %d\n", ethics->ethics_violations_count);
    printf("Safety Override Engaged: %s\n", ethics->safety_override_engaged ? "YES" : "NO");
    printf("No-Replication Rule Active: %s\n\n", ethics->no_replication_rule_active ? "YES" : "NO");
    
    printf("Safety Scores:\n");
    for (int i = 0; i < SAFETY_COUNT; i++) {
        printf("  Category %d: %.4f\n", i, ethics->safety_scores[i]);
    }
    
    printf("\nClarity Metrics:\n");
    for (int i = 0; i < 4; i++) {
        printf("  Metric %d: %.4f\n", i, ethics->clarity_metrics[i]);
    }
    
    printf("\nHuman Benefit Factors:\n");
    for (int i = 0; i < 3; i++) {
        printf("  Factor %d: %.4f\n", i, ethics->human_benefit_factors[i]);
    }
}

void ethics_log_violation(const ethics_monitor_t* ethics, const char* violation_type) {
    if (!ethics || !violation_type) return;
    
    printf("[ETHICS] Violation logged: %s\n", violation_type);
}
