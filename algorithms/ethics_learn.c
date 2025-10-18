#include "ethics_core.h"

#include <math.h>

static void clamp_weights_local(ethics_model_t* model) {
    if (model->weights.safety_weight < 0.1) model->weights.safety_weight = 0.1;
    if (model->weights.clarity_weight < 0.1) model->weights.clarity_weight = 0.1;
    if (model->weights.human_weight < 0.1) model->weights.human_weight = 0.1;
    if (model->weights.safety_weight > 2.0) model->weights.safety_weight = 2.0;
    if (model->weights.clarity_weight > 2.0) model->weights.clarity_weight = 2.0;
    if (model->weights.human_weight > 2.0) model->weights.human_weight = 2.0;
}

void ethics_learn_apply_feedback(ethics_model_t* model,
                                 double feedback,
                                 double learning_rate) {
    if (!model) return;
    if (!isfinite(feedback) || !isfinite(learning_rate)) return;

    const double clamp_lr = learning_rate < 0.0 ? 0.0 : (learning_rate > 1.0 ? 1.0 : learning_rate);

    double delta = clamp_lr * feedback;

    model->weights.safety_weight += delta * 0.5;
    model->weights.clarity_weight += delta * 0.3;
    model->weights.human_weight += delta * 0.2;

    clamp_weights_local(model);

    double adjust = clamp_lr * feedback * 0.05;
    model->thresholds.min_total += adjust;
    model->thresholds.min_total = fmax(1.0, fmin(2.5, model->thresholds.min_total));
}
