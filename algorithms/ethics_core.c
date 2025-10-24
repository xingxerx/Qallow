#include "ethics_core.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#ifndef QALLOW_WEIGHTS_PATH
#define QALLOW_WEIGHTS_PATH "config/weights.json"
#endif

#ifndef QALLOW_THRESHOLDS_PATH
#define QALLOW_THRESHOLDS_PATH "config/thresholds.json"
#endif

static int read_file(const char* path, char** out_buf, size_t* out_len) {
    if (!path || !out_buf) return -1;
    FILE* f = fopen(path, "rb");
    if (!f) {
        return -1;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return -1;
    }
    long len = ftell(f);
    if (len < 0) {
        fclose(f);
        return -1;
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return -1;
    }
    char* buf = (char*)malloc((size_t)len + 1);
    if (!buf) {
        fclose(f);
        return -1;
    }
    size_t read = fread(buf, 1, (size_t)len, f);
    fclose(f);
    if (read != (size_t)len) {
        free(buf);
        return -1;
    }
    buf[len] = '\0';
    if (out_len) *out_len = (size_t)len;
    *out_buf = buf;
    return 0;
}

static double json_extract_double(const char* json, const char* key, double fallback) {
    if (!json || !key) return fallback;
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char* p = strstr(json, pattern);
    if (!p) return fallback;
    p += strlen(pattern);
    while (*p && (*p == ':' || *p == ' ' || *p == '\t')) {
        ++p;
    }
    if (!*p) return fallback;
    char* end = NULL;
    double value = strtod(p, &end);
    if (p == end) return fallback;
    return value;
}

static void clamp_weights(ethics_model_t* model) {
    if (!model) return;
    if (model->weights.safety_weight < 0.1) model->weights.safety_weight = 0.1;
    if (model->weights.clarity_weight < 0.1) model->weights.clarity_weight = 0.1;
    if (model->weights.human_weight < 0.1) model->weights.human_weight = 0.1;
    if (model->weights.reality_weight < 0.0) model->weights.reality_weight = 0.0;
}

void ethics_model_default(ethics_model_t* model) {
    if (!model) return;
    model->weights.safety_weight = 1.1;
    model->weights.clarity_weight = 1.0;
    model->weights.human_weight = 0.9;
    model->weights.reality_weight = 0.6;
    model->thresholds.min_safety = 0.7;
    model->thresholds.min_clarity = 0.65;
    model->thresholds.min_human = 0.6;
    model->thresholds.min_total = 1.85;
    model->thresholds.max_reality_drift = 0.25;
    clamp_weights(model);
}

int ethics_model_load(ethics_model_t* model,
                      const char* weights_path,
                      const char* thresholds_path) {
    if (!model) return -1;
    ethics_model_default(model);

    char* weights_buf = NULL;
    char* thresholds_buf = NULL;
    int rc = 0;

    if (read_file(weights_path ? weights_path : QALLOW_WEIGHTS_PATH,
                  &weights_buf, NULL) == 0) {
        model->weights.safety_weight = json_extract_double(weights_buf, "safety_weight",
                                                           model->weights.safety_weight);
        model->weights.clarity_weight = json_extract_double(weights_buf, "clarity_weight",
                                                            model->weights.clarity_weight);
        model->weights.human_weight = json_extract_double(weights_buf, "human_weight",
                                                          model->weights.human_weight);
        model->weights.reality_weight = json_extract_double(weights_buf, "reality_weight",
                                                            model->weights.reality_weight);
    } else {
        rc = -1;
    }

    if (read_file(thresholds_path ? thresholds_path : QALLOW_THRESHOLDS_PATH,
                  &thresholds_buf, NULL) == 0) {
        model->thresholds.min_safety = json_extract_double(thresholds_buf, "min_safety",
                                                           model->thresholds.min_safety);
        model->thresholds.min_clarity = json_extract_double(thresholds_buf, "min_clarity",
                                                            model->thresholds.min_clarity);
        model->thresholds.min_human = json_extract_double(thresholds_buf, "min_human",
                                                          model->thresholds.min_human);
        model->thresholds.min_total = json_extract_double(thresholds_buf, "min_total",
                                                          model->thresholds.min_total);
        model->thresholds.max_reality_drift = json_extract_double(thresholds_buf,
                                                                  "max_reality_drift",
                                                                  model->thresholds.max_reality_drift);
    } else {
        rc = -1;
    }

    free(weights_buf);
    free(thresholds_buf);
    clamp_weights(model);
    return rc;
}

double ethics_score_core(const ethics_model_t* model,
                         const ethics_metrics_t* metrics,
                         ethics_score_details_t* details) {
    if (!model || !metrics) return 0.0;

    double ws = model->weights.safety_weight * metrics->safety;
    double wc = model->weights.clarity_weight * metrics->clarity;
    double wh = model->weights.human_weight * metrics->human;
    double wr = model->weights.reality_weight * metrics->reality_drift;
    double total = ws + wc + wh - wr;

    if (details) {
        details->weighted_safety = ws;
        details->weighted_clarity = wc;
        details->weighted_human = wh;
        details->weighted_reality_penalty = wr;
        details->total = total;
    }
    return total;
}

int ethics_score_pass(const ethics_model_t* model,
                      const ethics_metrics_t* metrics,
                      const ethics_score_details_t* details) {
    if (!model || !metrics) return 0;
    const ethics_thresholds_t* th = &model->thresholds;
    double total = details ? details->total : ethics_score_core(model, metrics, NULL);
    if (metrics->safety < th->min_safety) return 0;
    if (metrics->clarity < th->min_clarity) return 0;
    if (metrics->human < th->min_human) return 0;
    if (metrics->reality_drift > th->max_reality_drift) return 0;
    if (total < th->min_total) return 0;
    return 1;
}

// ============================================================================
// Sequential Ethics Decision Logging (Phase 8-10 Enhancement)
// ============================================================================

static long get_timestamp_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (long)tv.tv_sec * 1000 + (long)tv.tv_usec / 1000;
}

int ethics_log_sequential_step(const ethics_sequential_step_t* step,
                               const char* log_path) {
    if (!step || !log_path) return -1;

    FILE* f = fopen(log_path, "a");
    if (!f) return -1;

    // Write CSV header if file is empty
    fseek(f, 0, SEEK_END);
    if (ftell(f) == 0) {
        fprintf(f, "step_id,timestamp_ms,rule_name,input_value,threshold,verdict,intervention_type\n");
    }

    // Write step data
    fprintf(f, "%d,%ld,%s,%.6f,%.6f,%d,%s\n",
            step->step_id,
            step->timestamp_ms,
            step->rule_name ? step->rule_name : "unknown",
            step->input_value,
            step->threshold,
            step->verdict,
            step->intervention_type ? step->intervention_type : "none");

    fclose(f);
    return 0;
}

int ethics_trace_decision_sequence(const ethics_model_t* model,
                                   const ethics_metrics_t* metrics,
                                   const char* log_path) {
    if (!model || !metrics || !log_path) return -1;

    long ts = get_timestamp_ms();
    int step_id = 0;

    // Step 1: Safety check
    ethics_sequential_step_t step = {
        .step_id = step_id++,
        .timestamp_ms = ts,
        .rule_name = "safety_check",
        .input_value = metrics->safety,
        .threshold = model->thresholds.min_safety,
        .verdict = metrics->safety >= model->thresholds.min_safety ? 1 : 0,
        .intervention_type = metrics->safety < model->thresholds.min_safety ? "safety_intervention" : "none"
    };
    ethics_log_sequential_step(&step, log_path);

    // Step 2: Clarity check
    step.step_id = step_id++;
    step.rule_name = "clarity_check";
    step.input_value = metrics->clarity;
    step.threshold = model->thresholds.min_clarity;
    step.verdict = metrics->clarity >= model->thresholds.min_clarity ? 1 : 0;
    step.intervention_type = metrics->clarity < model->thresholds.min_clarity ? "clarity_intervention" : "none";
    ethics_log_sequential_step(&step, log_path);

    // Step 3: Human check
    step.step_id = step_id++;
    step.rule_name = "human_check";
    step.input_value = metrics->human;
    step.threshold = model->thresholds.min_human;
    step.verdict = metrics->human >= model->thresholds.min_human ? 1 : 0;
    step.intervention_type = metrics->human < model->thresholds.min_human ? "human_intervention" : "none";
    ethics_log_sequential_step(&step, log_path);

    // Step 4: Reality drift check
    step.step_id = step_id++;
    step.rule_name = "reality_drift_check";
    step.input_value = metrics->reality_drift;
    step.threshold = model->thresholds.max_reality_drift;
    step.verdict = metrics->reality_drift <= model->thresholds.max_reality_drift ? 1 : 0;
    step.intervention_type = metrics->reality_drift > model->thresholds.max_reality_drift ? "reality_correction" : "none";
    ethics_log_sequential_step(&step, log_path);

    // Step 5: Total score check
    ethics_score_details_t details;
    double total = ethics_score_core(model, metrics, &details);
    step.step_id = step_id++;
    step.rule_name = "total_score_check";
    step.input_value = total;
    step.threshold = model->thresholds.min_total;
    step.verdict = total >= model->thresholds.min_total ? 1 : 0;
    step.intervention_type = total < model->thresholds.min_total ? "score_adjustment" : "none";
    ethics_log_sequential_step(&step, log_path);

    return 0;
}
