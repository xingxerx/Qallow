#include "ethics_core.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
}

void ethics_model_default(ethics_model_t* model) {
    if (!model) return;
    model->weights.safety_weight = 1.1;
    model->weights.clarity_weight = 1.0;
    model->weights.human_weight = 0.9;
    model->thresholds.min_safety = 0.7;
    model->thresholds.min_clarity = 0.65;
    model->thresholds.min_human = 0.6;
    model->thresholds.min_total = 1.85;
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
    double total = ws + wc + wh;

    if (details) {
        details->weighted_safety = ws;
        details->weighted_clarity = wc;
        details->weighted_human = wh;
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
    if (total < th->min_total) return 0;
    return 1;
}
