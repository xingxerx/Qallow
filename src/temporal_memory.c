/* src/temporal_memory.c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "temporal_memory.h"

static float dot_product(const float *a, const float *b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) sum += a[i] * b[i];
    return sum;
}

static float vector_norm(const float *v, size_t dim) {
    return sqrtf(dot_product(v, v, dim));
}

static float cosine_similarity(const float *a, const float *b, size_t dim) {
    float dot = dot_product(a, b, dim);
    float norm_a = vector_norm(a, dim);
    float norm_b = vector_norm(b, dim);
    return (norm_a > 0 && norm_b > 0) ? dot / (norm_a * norm_b) : 0.0f;
}

int tm_init(TemporalMemory *tm, float drift_threshold) {
    if (!tm) return -1;

    tm->episodes = calloc(TM_EPISODIC_SIZE, sizeof(MemoryVector));
    tm->semantics = calloc(TM_SEMANTIC_SIZE, sizeof(MemoryVector));
    tm->gradient_window = 64;  // default history window
    tm->gradient_history = calloc(tm->gradient_window, sizeof(float));

    if (!tm->episodes || !tm->semantics || !tm->gradient_history) {
        tm_free(tm);
        return -2;
    }

    tm->episode_head = 0;
    tm->episode_count = 0;
    tm->semantic_count = 0;
    tm->drift_threshold = drift_threshold;
    tm->avg_coherence = 1.0f;
    tm->coherence_checks = 0;
    tm->gradient_index = 0;

    return 0;
}

void tm_free(TemporalMemory *tm) {
    if (!tm) return;
    free(tm->episodes);
    free(tm->semantics);
    free(tm->gradient_history);
    memset(tm, 0, sizeof(*tm));
}

int tm_store_episodic(TemporalMemory *tm, const float *vec, size_t dim) {
    if (!tm || !vec || dim > TM_VECTOR_DIM) return -1;

    size_t insert_idx = tm->episode_head;
    MemoryVector *ep = &tm->episodes[insert_idx];
    memcpy(ep->vec, vec, dim * sizeof(float));
    if (dim < TM_VECTOR_DIM) {
        memset(ep->vec + dim, 0, (TM_VECTOR_DIM - dim) * sizeof(float));
    }
    ep->timestamp = time(NULL);
    ep->access_count = 1;
    ep->label[0] = '\0';

    tm->episode_head = (tm->episode_head + 1) % TM_EPISODIC_SIZE;
    if (tm->episode_count < TM_EPISODIC_SIZE) tm->episode_count++;

    ep->coherence_score = tm_check_coherence(tm, vec, dim);
    return 0;
}

int tm_store_semantic(TemporalMemory *tm, const float *vec, size_t dim, const char *label) {
    if (!tm || !vec || dim > TM_VECTOR_DIM || tm->semantic_count >= TM_SEMANTIC_SIZE) return -1;

    MemoryVector *sem = &tm->semantics[tm->semantic_count];
    memcpy(sem->vec, vec, dim * sizeof(float));
    if (dim < TM_VECTOR_DIM) {
        memset(sem->vec + dim, 0, (TM_VECTOR_DIM - dim) * sizeof(float));
    }
    sem->timestamp = time(NULL);
    sem->access_count = 1;
    sem->coherence_score = 1.0f;
    if (label && *label) {
        strncpy(sem->label, label, sizeof(sem->label) - 1);
        sem->label[sizeof(sem->label) - 1] = '\0';
    } else {
        sem->label[0] = '\0';
    }

    tm->semantic_count++;

    return 0;
}

int tm_retrieve_similar(TemporalMemory *tm, const float *query, size_t dim, MemoryVector **result) {
    if (!tm || !query || !result || dim > TM_VECTOR_DIM) return -1;

    float best_sim = -1.0f;
    MemoryVector *best = NULL;

    size_t episodic_to_check = (tm->episode_count < 16) ? tm->episode_count : 16;
    for (size_t i = 0; i < episodic_to_check; ++i) {
        size_t idx = (tm->episode_head + TM_EPISODIC_SIZE - i - 1) % TM_EPISODIC_SIZE;
        float sim = cosine_similarity(query, tm->episodes[idx].vec, dim);
        if (sim > best_sim) {
            best_sim = sim;
            best = &tm->episodes[idx];
        }
    }

    size_t semantic_to_check = (tm->semantic_count < 32) ? tm->semantic_count : 32;
    for (size_t i = 0; i < semantic_to_check; ++i) {
        float sim = cosine_similarity(query, tm->semantics[i].vec, dim);
        if (sim > best_sim) {
            best_sim = sim;
            best = &tm->semantics[i];
        }
    }

    if (best) {
        best->access_count++;
        *result = best;
        return 0;
    }

    *result = NULL;
    return 1; // No similar memory found
}

float tm_check_coherence(TemporalMemory *tm, const float *vec, size_t dim) {
    if (!tm || !vec || dim > TM_VECTOR_DIM) return 0.0f; // Invalid input

    float total_sim = 0.0f;
    size_t count = 0;

    size_t episodic_to_check = (tm->episode_count < 10) ? tm->episode_count : 10;
    for (size_t i = 0; i < episodic_to_check; ++i) {
        size_t idx = (tm->episode_head + TM_EPISODIC_SIZE - i - 1) % TM_EPISODIC_SIZE;
        total_sim += cosine_similarity(vec, tm->episodes[idx].vec, dim);
        count++;
    }

    size_t semantic_to_check = (tm->semantic_count < 5) ? tm->semantic_count : 5;
    for (size_t i = 0; i < semantic_to_check; ++i) {
        total_sim += cosine_similarity(vec, tm->semantics[i].vec, dim);
        count++;
    }

    float current_coherence = (count > 0) ? total_sim / count : 1.0f;

    // Update average coherence
    if (tm->coherence_checks == 0) {
        tm->avg_coherence = current_coherence;
    } else {
        tm->avg_coherence = (tm->avg_coherence * tm->coherence_checks + current_coherence) / (tm->coherence_checks + 1);
    }
    tm->coherence_checks++;

    return current_coherence;
}

int tm_audit_drift(TemporalMemory *tm, float *drift_report) {
    if (!tm || !drift_report) return -1;

    float recent_coherence = 0.0f;
    size_t recent_count = (tm->episode_count < 20) ? tm->episode_count : 20;

    for (size_t i = 0; i < recent_count; ++i) {
        size_t idx = (tm->episode_head + TM_EPISODIC_SIZE - i - 1) % TM_EPISODIC_SIZE;
        recent_coherence += tm->episodes[idx].coherence_score;
    }
    if (recent_count > 0) {
        recent_coherence /= recent_count;
    } else {
        recent_coherence = tm->avg_coherence;
    }

    *drift_report = tm->avg_coherence - recent_coherence;

    if (fabsf(*drift_report) > tm->drift_threshold) {
        printf("[TM] Drift detected: %.3f (threshold: %.3f)\n", *drift_report, tm->drift_threshold);
        return 1;
    }

    return 0;
}

int tm_recalibrate(TemporalMemory *tm) {
    if (!tm) return -1;

    printf("[TM] Recalibrating memory coherence...\n");

    for (size_t i = 0; i < tm->episode_count; ++i) {
        size_t idx = (tm->episode_head + TM_EPISODIC_SIZE - i - 1) % TM_EPISODIC_SIZE;
        tm->episodes[idx].coherence_score =
            tm_check_coherence(tm, tm->episodes[idx].vec, TM_VECTOR_DIM);
    }

    size_t semantic_to_check = tm->semantic_count;
    for (size_t i = 0; i < semantic_to_check; ++i) {
        tm->semantics[i].coherence_score =
            tm_check_coherence(tm, tm->semantics[i].vec, TM_VECTOR_DIM);
    }

    for (size_t i = 0; i < tm->episode_count; ++i) {
        size_t idx = (tm->episode_head + TM_EPISODIC_SIZE - i - 1) % TM_EPISODIC_SIZE;
        if (tm->episodes[idx].access_count > 10 &&
            tm->semantic_count < TM_SEMANTIC_SIZE) {
            tm_store_semantic(tm, tm->episodes[idx].vec, TM_VECTOR_DIM, "promoted");
        }
    }

    return 0;
}

int tm_update_gradient(TemporalMemory *tm, float gradient) {
    if (!tm || !tm->gradient_history || tm->gradient_window == 0) return -1;

    tm->gradient_history[tm->gradient_index % tm->gradient_window] = gradient;
    tm->gradient_index++;

    return 0;
}

float tm_predict_next(const TemporalMemory *tm) {
    if (!tm || !tm->gradient_history || tm->gradient_window == 0) return 0.0f;

    float weighted_sum = 0.0f;
    float weight_total = 0.0f;

    size_t count = (tm->gradient_index < tm->gradient_window)
                       ? tm->gradient_index
                       : tm->gradient_window;
    if (count == 0) return 0.0f;

    for (size_t i = 0; i < count; ++i) {
        size_t idx = (tm->gradient_index + tm->gradient_window - i - 1) % tm->gradient_window;
        float weight = (float)(i + 1) / count;
        weighted_sum += tm->gradient_history[idx] * weight;
        weight_total += weight;
    }

    if (weight_total == 0.0f) {
        return 0.0f;
    }
    return weighted_sum / weight_total;
}
