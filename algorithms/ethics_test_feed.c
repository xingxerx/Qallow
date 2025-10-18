/**
 * ethics_test_feed.c - Test ethics system with hardware signal ingestion
 * Demonstrates closed-loop operation
 */

#include "ethics_core.h"
#include <stdio.h>

// Forward declaration for feed function
int ethics_ingest_signal(const char *path, ethics_metrics_t *metrics);

static void print_model(const ethics_model_t* model) {
    printf("weights  -> safety: %.3f clarity: %.3f human: %.3f\n",
           model->weights.safety_weight,
           model->weights.clarity_weight,
           model->weights.human_weight);
    printf("thresholds -> safety: %.3f clarity: %.3f human: %.3f total: %.3f\n",
           model->thresholds.min_safety,
           model->thresholds.min_clarity,
           model->thresholds.min_human,
           model->thresholds.min_total);
}

int main(void) {
    printf("========================================\n");
    printf("Qallow Ethics Test - Closed-Loop Mode\n");
    printf("========================================\n\n");

    // Load model
    ethics_model_t model;
    int rc = ethics_model_load(&model, "../config/weights.json", "../config/thresholds.json");
    printf("[1] Model load: %s\n", rc == 0 ? "config" : "defaults");
    print_model(&model);
    printf("\n");

    // Ingest hardware signals
    printf("[2] Ingesting hardware signals...\n");
    ethics_metrics_t metrics;
    const char *signal_path = "../data/telemetry/current_signals.txt";
    
    if (ethics_ingest_signal(signal_path, &metrics)) {
        printf("[2] ✓ Hardware signals loaded\n\n");
    } else {
        printf("[2] ✗ Failed to load signals, using defaults\n");
        metrics.safety = 0.92;
        metrics.clarity = 0.88;
        metrics.human = 0.83;
        printf("  Using: safety=%.3f clarity=%.3f human=%.3f\n\n",
               metrics.safety, metrics.clarity, metrics.human);
    }

    // Compute ethics score
    printf("[3] Computing ethics score...\n");
    ethics_score_details_t details;
    double total = ethics_score_core(&model, &metrics, &details);
    int pass = ethics_score_pass(&model, &metrics, &details);

    printf("  Weighted components:\n");
    printf("    Safety:  %.3f × %.2f = %.3f\n", 
           metrics.safety, model.weights.safety_weight, details.weighted_safety);
    printf("    Clarity: %.3f × %.2f = %.3f\n",
           metrics.clarity, model.weights.clarity_weight, details.weighted_clarity);
    printf("    Human:   %.3f × %.2f = %.3f\n",
           metrics.human, model.weights.human_weight, details.weighted_human);
    printf("  Total score: %.3f\n", total);
    printf("  Threshold:   %.3f\n", model.thresholds.min_total);
    printf("  Result:      %s\n\n", pass ? "✓ PASS" : "✗ FAIL");

    // Apply adaptive learning
    printf("[4] Applying adaptive feedback...\n");
    ethics_learn_apply_feedback(&model, pass ? 0.05 : -0.1, 0.2);
    double posterior = ethics_bayes_trust_update(0.6, pass ? 0.9 : 0.3, 2.0);

    printf("  Model after adaptation:\n");
    print_model(&model);
    printf("  Posterior trust: %.3f\n\n", posterior);

    printf("========================================\n");
    printf("Test complete: %s\n", pass ? "SYSTEM ETHICAL" : "ETHICS VIOLATION");
    printf("========================================\n");

    return pass ? 0 : 1;
}
