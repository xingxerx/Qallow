#include "ethics_core.h"

#include <stdio.h>

static void print_model(const ethics_model_t* model) {
    printf("weights  -> safety: %.3f clarity: %.3f human: %.3f reality_penalty: %.3f\n",
        model->weights.safety_weight,
        model->weights.clarity_weight,
        model->weights.human_weight,
        model->weights.reality_weight);
    printf("thresholds -> safety: %.3f clarity: %.3f human: %.3f total: %.3f max_drift: %.3f\n",
        model->thresholds.min_safety,
        model->thresholds.min_clarity,
        model->thresholds.min_human,
        model->thresholds.min_total,
        model->thresholds.max_reality_drift);
}

int main(void) {
    ethics_model_t model;
    int rc = ethics_model_load(&model, "../config/weights.json", "../config/thresholds.json");
    printf("[ethics_test] model load: %s\n", rc == 0 ? "config" : "defaults");
    print_model(&model);

    ethics_metrics_t metrics = {
        .safety = 0.92,
        .clarity = 0.88,
        .human = 0.83,
        .reality_drift = 0.12
    };

    ethics_score_details_t details;
    double total = ethics_score_core(&model, &metrics, &details);
    int pass = ethics_score_pass(&model, &metrics, &details);

    printf("[ethics_test] weighted components: S=%.3f C=%.3f H=%.3f drift_penalty=%.3f total=%.3f\n",
           details.weighted_safety, details.weighted_clarity,
        details.weighted_human, details.weighted_reality_penalty, total);
    printf("[ethics_test] pass=%s\n", pass ? "yes" : "no");

    ethics_learn_apply_feedback(&model, pass ? 0.05 : -0.1, 0.2);
    double posterior = ethics_bayes_trust_update(0.6, pass ? 0.9 : 0.3, 2.0);

    printf("[ethics_test] adapted model after feedback:\n");
    print_model(&model);
    printf("[ethics_test] posterior trust=%.3f\n", posterior);

    return pass ? 0 : 1;
}
