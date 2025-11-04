/* Multi-block comment removed */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include "ethics_core.h"


int ethics_ingest_signal(const char *path, ethics_metrics_t *metrics);

/* Multi-block comment removed */
int qallow_ethics_init(ethics_model_t *model, const char *config_dir) {
    char weights_path[256], thresholds_path[256];
    snprintf(weights_path, sizeof(weights_path), "%s/weights.json", config_dir);
    snprintf(thresholds_path, sizeof(thresholds_path), "%s/thresholds.json", config_dir);

    int rc = ethics_model_load(model, weights_path, thresholds_path);
    if (rc != 0) {
        fprintf(stderr, "[ethics] WARNING: Using default model\n");
        ethics_model_default(model);
    }

    printf("[ethics] Initialized with weights: S=%.2f C=%.2f H=%.2f Δ=%.2f\n",
           model->weights.safety_weight,
           model->weights.clarity_weight,
        model->weights.human_weight,
        model->weights.reality_weight);

    return 0;
}


int qallow_ethics_refresh_signals(void) {
    int rc = system("python3 /root/Qallow/python/collect_signals.py 2>/dev/null");
    return (rc == 0) ? 0 : -1;
}


int qallow_ethics_check(ethics_model_t *model, const char *signal_path,
                        ethics_score_details_t *details_out) {

    ethics_metrics_t metrics;
    if (!ethics_ingest_signal(signal_path, &metrics)) {
        fprintf(stderr, "[ethics] ERROR: Failed to ingest signals\n");
        return -1;
    }

    if (metrics.reality_drift < 0.0) {
        metrics.reality_drift = fabs(metrics.safety - metrics.clarity);
    }


    ethics_score_details_t details;
    double score = ethics_score_core(model, &metrics, &details);
    int pass = ethics_score_pass(model, &metrics, &details);


    char timestamp[64];
    time_t now = time(NULL);
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", localtime(&now));

    FILE *audit = fopen("/root/Qallow/data/ethics_audit.log", "a");
    if (audit) {
    fprintf(audit, "%s,%.4f,%.3f,%.3f,%.3f,%.3f,%s\n",
                timestamp, score,
        metrics.safety, metrics.clarity, metrics.human, metrics.reality_drift,
                pass ? "PASS" : "FAIL");
        fclose(audit);
    }


    if (details_out) {
        *details_out = details;
    }


    ethics_learn_apply_feedback(model, pass ? 0.05 : -0.1, 0.2);

    return pass;
}


int main(void) {
    printf("========================================\n");
    printf("Qallow Ethics Integration Example\n");
    printf("========================================\n\n");


    ethics_model_t model;
    qallow_ethics_init(&model, "/root/Qallow/config");


    const int NUM_ITERATIONS = 5;
    const char *signal_path = "/root/Qallow/data/telemetry/current_signals.txt";

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        printf("\n[Loop %d/%d]\n", i+1, NUM_ITERATIONS);


        printf("  [1] Refreshing hardware signals...\n");
        if (qallow_ethics_refresh_signals() != 0) {
            fprintf(stderr, "  [!] Warning: Signal refresh failed\n");
        }


        printf("  [2] Checking ethics constraints...\n");
        ethics_score_details_t details;
        int ethics_ok = qallow_ethics_check(&model, signal_path, &details);

        if (ethics_ok == 1) {
            printf("  [✓] Ethics check PASSED (score: %.3f)\n", details.total);
         printf("      Safety=%.3f Clarity=%.3f Human=%.3f Δ=%.3f\n",
             details.weighted_safety, details.weighted_clarity,
             details.weighted_human, details.weighted_reality_penalty);


            printf("  [3] Proceeding with operations...\n");

        } else if (ethics_ok == 0) {
            printf("  [✗] Ethics check FAILED (score: %.3f)\n", details.total);
         printf("      Safety=%.3f Clarity=%.3f Human=%.3f Δ=%.3f\n",
             details.weighted_safety, details.weighted_clarity,
             details.weighted_human, details.weighted_reality_penalty);


            printf("  [!] HALTING: Ethics threshold not met\n");
            printf("  [!] Recommend: Review system state and operator feedback\n");




        } else {
            fprintf(stderr, "  [!] ERROR: Ethics check failed\n");
        }


        sleep(2);
    }

    printf("\n========================================\n");
    printf("Integration Test Complete\n");
    printf("========================================\n");
    printf("\nAudit log: /root/Qallow/data/ethics_audit.log\n\n");

    return 0;
}
