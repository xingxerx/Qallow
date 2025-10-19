#include "ethics_core.h"

#include <assert.h>
#include <stdio.h>

static void test_ethics_defaults(void) {
    ethics_model_t model;
    ethics_model_default(&model);

    assert(model.weights.safety_weight >= 0.1);
    assert(model.thresholds.min_total > 1.0);
}

static void test_ethics_pass_fail(void) {
    ethics_model_t model;
    ethics_model_default(&model);

    ethics_metrics_t metrics = {
        .safety = 0.9,
        .clarity = 0.9,
        .human = 0.9,
    };

    ethics_score_details_t details;
    double total = ethics_score_core(&model, &metrics, &details);
    assert(total > 0.0);
    assert(ethics_score_pass(&model, &metrics, &details) == 1);

    metrics.human = 0.1;
    total = ethics_score_core(&model, &metrics, &details);
    assert(total > 0.0);
    assert(ethics_score_pass(&model, &metrics, &details) == 0);
}

int main(void) {
    test_ethics_defaults();
    test_ethics_pass_fail();
    puts("ethics_core unit tests passed");
    return 0;
}
