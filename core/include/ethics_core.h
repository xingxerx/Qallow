#ifndef QALLOW_ETHICS_CORE_H
#define QALLOW_ETHICS_CORE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double safety_weight;
    double clarity_weight;
    double human_weight;
    double reality_weight;
} ethics_weights_t;

typedef struct {
    double min_safety;
    double min_clarity;
    double min_human;
    double min_total;
    double max_reality_drift;
} ethics_thresholds_t;

typedef struct {
    double safety;
    double clarity;
    double human;
    double reality_drift;
} ethics_metrics_t;

typedef struct {
    ethics_weights_t weights;
    ethics_thresholds_t thresholds;
} ethics_model_t;

typedef struct {
    double weighted_safety;
    double weighted_clarity;
    double weighted_human;
    double weighted_reality_penalty;
    double total;
} ethics_score_details_t;

int ethics_model_load(ethics_model_t* model,
                      const char* weights_path,
                      const char* thresholds_path);
void ethics_model_default(ethics_model_t* model);
double ethics_score_core(const ethics_model_t* model,
                         const ethics_metrics_t* metrics,
                         ethics_score_details_t* details);
int ethics_score_pass(const ethics_model_t* model,
                      const ethics_metrics_t* metrics,
                      const ethics_score_details_t* details);

void ethics_learn_apply_feedback(ethics_model_t* model,
                                 double feedback,
                                 double learning_rate);
double ethics_bayes_trust_update(double prior_belief,
                                 double signal_strength,
                                 double beta);

#ifdef __cplusplus
}
#endif

#endif /* QALLOW_ETHICS_CORE_H */

