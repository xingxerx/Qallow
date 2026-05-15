#include "qallow/temporal_memory.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================================
 * Helper: Vector operations
 * ============================================================================ */

static float qallow_cosine_similarity(const float* a, const float* b, int dim) {
    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (int i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    float denom = sqrtf(norm_a) * sqrtf(norm_b);
    if (denom < 1e-10f) return 0.0f;

    return dot / denom;
}

static void qallow_normalize_vector(float* v, int dim) {
    float norm = 0.0f;
    for (int i = 0; i < dim; ++i) {
        norm += v[i] * v[i];
    }
    norm = sqrtf(norm);
    if (norm < 1e-10f) return;

    for (int i = 0; i < dim; ++i) {
        v[i] /= norm;
    }
}

/* ============================================================================
 * Initialize & Cleanup
 * ============================================================================ */

qallow_temporal_memory_t qallow_temporal_memory_init(int max_events) {
    qallow_temporal_memory_t mem;
    memset(&mem, 0, sizeof(mem));

    mem.event_capacity = max_events > 0 ? max_events : 1000;
    mem.events = (qallow_episodic_event_t*)malloc(
        sizeof(qallow_episodic_event_t) * mem.event_capacity);

    mem.embedding_capacity = 100;
    mem.embeddings = (qallow_semantic_embedding_t*)malloc(
        sizeof(qallow_semantic_embedding_t) * mem.embedding_capacity);

    mem.coherence_threshold = 0.1f;
    mem.next_event_id = 1;
    mem.drift_history_idx = 0;

    /* Initialize baseline coherence to neutral */
    mem.baseline_coherence.entropy_score = 0.5f;
    mem.baseline_coherence.harmony_score = 0.8f;
    mem.baseline_coherence.temporal_gradient = 0.0f;
    mem.baseline_coherence.baseline_deviation = 0.0f;

    return mem;
}

void qallow_temporal_memory_free(qallow_temporal_memory_t* mem) {
    if (!mem) return;

    if (mem->events) {
        free(mem->events);
        mem->events = NULL;
    }

    if (mem->embeddings) {
        free(mem->embeddings);
        mem->embeddings = NULL;
    }

    mem->event_count = 0;
    mem->embedding_count = 0;
}

/* ============================================================================
 * Level 1: Episodic Events
 * ============================================================================ */

uint64_t qallow_temporal_memory_log_event(qallow_temporal_memory_t* mem,
                                          const char* event_type,
                                          const char* description,
                                          float score) {
    if (!mem || !event_type) return 0;

    /* Expand buffer if needed */
    if (mem->event_count >= mem->event_capacity) {
        mem->event_capacity *= 2;
        mem->events = (qallow_episodic_event_t*)realloc(
            mem->events, sizeof(qallow_episodic_event_t) * mem->event_capacity);
    }

    /* Add event */
    qallow_episodic_event_t* event = &mem->events[mem->event_count++];
    event->event_id = mem->next_event_id++;
    event->timestamp = time(NULL);
    strncpy(event->event_type, event_type, sizeof(event->event_type) - 1);
    event->event_type[sizeof(event->event_type) - 1] = '\0';

    if (description) {
        strncpy(event->event_description, description,
                sizeof(event->event_description) - 1);
        event->event_description[sizeof(event->event_description) - 1] = '\0';
    }

    event->event_score = fminf(1.0f, fmaxf(0.0f, score));

    return event->event_id;
}

qallow_episodic_event_t* qallow_temporal_memory_query_events(
    qallow_temporal_memory_t* mem,
    const char* event_type,
    int max_results) {

    if (!mem || max_results <= 0) return NULL;

    qallow_episodic_event_t* results = (qallow_episodic_event_t*)malloc(
        sizeof(qallow_episodic_event_t) * max_results);
    if (!results) return NULL;

    int result_count = 0;

    /* Query in reverse order (most recent first) */
    for (int i = mem->event_count - 1; i >= 0 && result_count < max_results; --i) {
        if (!event_type || strcmp(mem->events[i].event_type, event_type) == 0) {
            results[result_count++] = mem->events[i];
        }
    }

    return results;
}

void qallow_temporal_memory_prune_events(qallow_temporal_memory_t* mem,
                                         int keep_count) {
    if (!mem || keep_count < 0) return;

    if (mem->event_count > keep_count) {
        /* Remove oldest events, keep most recent */
        int remove_count = mem->event_count - keep_count;
        memmove(mem->events,
                &mem->events[remove_count],
                sizeof(qallow_episodic_event_t) * keep_count);
        mem->event_count = keep_count;
    }
}

/* ============================================================================
 * Level 2: Semantic Embeddings
 * ============================================================================ */

int qallow_temporal_memory_add_embedding(qallow_temporal_memory_t* mem,
                                         const char* concept_name,
                                         const float embedding[128],
                                         float confidence) {
    if (!mem || !concept_name || !embedding) return -1;

    /* Check if exists (update) */
    for (int i = 0; i < mem->embedding_count; ++i) {
        if (strcmp(mem->embeddings[i].concept_name, concept_name) == 0) {
            memcpy(mem->embeddings[i].embedding, embedding, sizeof(float) * 128);
            mem->embeddings[i].confidence = confidence;
            mem->embeddings[i].occurrence_count++;
            return 0;
        }
    }

    /* Expand if needed */
    if (mem->embedding_count >= mem->embedding_capacity) {
        mem->embedding_capacity *= 2;
        mem->embeddings = (qallow_semantic_embedding_t*)realloc(
            mem->embeddings,
            sizeof(qallow_semantic_embedding_t) * mem->embedding_capacity);
    }

    /* Add new embedding */
    qallow_semantic_embedding_t* emb = &mem->embeddings[mem->embedding_count++];
    strncpy(emb->concept_name, concept_name, sizeof(emb->concept_name) - 1);
    emb->concept_name[sizeof(emb->concept_name) - 1] = '\0';
    memcpy(emb->embedding, embedding, sizeof(float) * 128);
    emb->confidence = confidence;
    emb->occurrence_count = 1;

    return 0;
}

qallow_semantic_embedding_t* qallow_temporal_memory_get_embedding(
    qallow_temporal_memory_t* mem,
    const char* concept_name) {

    if (!mem || !concept_name) return NULL;

    for (int i = 0; i < mem->embedding_count; ++i) {
        if (strcmp(mem->embeddings[i].concept_name, concept_name) == 0) {
            return &mem->embeddings[i];
        }
    }

    return NULL;
}

qallow_semantic_embedding_t* qallow_temporal_memory_nearest_embeddings(
    qallow_temporal_memory_t* mem,
    const float query_embedding[128],
    int max_results) {

    if (!mem || !query_embedding || max_results <= 0) return NULL;

    /* Calculate similarities */
    typedef struct {
        int idx;
        float similarity;
    } result_t;

    result_t* scores = (result_t*)malloc(sizeof(result_t) * mem->embedding_count);
    if (!scores) return NULL;

    for (int i = 0; i < mem->embedding_count; ++i) {
        scores[i].idx = i;
        scores[i].similarity = qallow_cosine_similarity(
            query_embedding, mem->embeddings[i].embedding, 128);
    }

    /* Sort by similarity (descending) */
    for (int i = 0; i < mem->embedding_count - 1; ++i) {
        for (int j = i + 1; j < mem->embedding_count; ++j) {
            if (scores[j].similarity > scores[i].similarity) {
                result_t tmp = scores[i];
                scores[i] = scores[j];
                scores[j] = tmp;
            }
        }
    }

    /* Collect results */
    int result_count = mem->embedding_count < max_results ? mem->embedding_count : max_results;
    qallow_semantic_embedding_t* results = (qallow_semantic_embedding_t*)malloc(
        sizeof(qallow_semantic_embedding_t) * (result_count + 1));

    if (results) {
        for (int i = 0; i < result_count; ++i) {
            results[i] = mem->embeddings[scores[i].idx];
        }
        results[result_count].concept_name[0] = '\0';  /* Null terminator */
    }

    free(scores);
    return results;
}

/* ============================================================================
 * Level 3: Coherence & Drift
 * ============================================================================ */

void qallow_temporal_memory_update_coherence(qallow_temporal_memory_t* mem,
                                             float temporal_gradient,
                                             float baseline_deviation,
                                             float entropy_score,
                                             float harmony_score) {
    if (!mem) return;

    mem->current_coherence.temporal_gradient = temporal_gradient;
    mem->current_coherence.baseline_deviation = baseline_deviation;
    mem->current_coherence.entropy_score = fminf(1.0f, fmaxf(0.0f, entropy_score));
    mem->current_coherence.harmony_score = fminf(1.0f, fmaxf(0.0f, harmony_score));
    mem->current_coherence.last_baseline_update = time(NULL);
}

float qallow_temporal_memory_detect_drift(qallow_temporal_memory_t* mem) {
    if (!mem) return 0.0f;

    /* Calculate drift as L2 distance from baseline */
    float entropy_diff = mem->current_coherence.entropy_score - mem->baseline_coherence.entropy_score;
    float harmony_diff = mem->current_coherence.harmony_score - mem->baseline_coherence.harmony_score;
    float gradient_diff = mem->current_coherence.temporal_gradient - mem->baseline_coherence.temporal_gradient;

    float drift = sqrtf(entropy_diff * entropy_diff +
                       harmony_diff * harmony_diff +
                       gradient_diff * gradient_diff) / sqrtf(3.0f);

    drift = fminf(1.0f, fmaxf(0.0f, drift));

    /* Record in history */
    mem->drift_history[mem->drift_history_idx++] = drift;
    if (mem->drift_history_idx >= 100) {
        mem->drift_history_idx = 0;
    }

    /* Log alert if exceeds threshold */
    if (drift > mem->coherence_threshold) {
        fprintf(stderr, "[DRIFT] Alert: drift=%.3f exceeds threshold=%.3f\n",
                drift, mem->coherence_threshold);
    }

    return drift;
}

void qallow_temporal_memory_set_baseline(qallow_temporal_memory_t* mem) {
    if (!mem) return;
    mem->baseline_coherence = mem->current_coherence;
    fprintf(stderr, "[MEMORY] Baseline coherence set (entropy=%.3f, harmony=%.3f)\n",
            mem->baseline_coherence.entropy_score, mem->baseline_coherence.harmony_score);
}

qallow_coherence_metrics_t qallow_temporal_memory_get_coherence(
    const qallow_temporal_memory_t* mem) {

    qallow_coherence_metrics_t blank;
    memset(&blank, 0, sizeof(blank));

    if (!mem) return blank;
    return mem->current_coherence;
}

void qallow_temporal_memory_set_drift_threshold(qallow_temporal_memory_t* mem,
                                                float threshold) {
    if (!mem) return;
    mem->coherence_threshold = fminf(1.0f, fmaxf(0.0f, threshold));
}

/* ============================================================================
 * Drift Statistics
 * ============================================================================ */

float qallow_temporal_memory_get_average_drift(const qallow_temporal_memory_t* mem,
                                               int window_size) {
    if (!mem || window_size <= 0) return 0.0f;

    float sum = 0.0f;
    int count = 0;

    for (int i = 0; i < 100 && i < window_size; ++i) {
        sum += mem->drift_history[i];
        count++;
    }

    return count > 0 ? sum / count : 0.0f;
}

int qallow_temporal_memory_get_drift_trend(const qallow_temporal_memory_t* mem) {
    if (!mem) return 0;

    float recent_avg = qallow_temporal_memory_get_average_drift(mem, 20);
    float older_avg = qallow_temporal_memory_get_average_drift(mem, 50);

    float diff = recent_avg - older_avg;
    if (diff > 0.01f) return 1;   /* Increasing */
    if (diff < -0.01f) return -1; /* Decreasing */
    return 0;                      /* Stable */
}

int qallow_temporal_memory_export_json(const qallow_temporal_memory_t* mem,
                                        const char* json_path) {
    if (!mem || !json_path) return -1;

    FILE* f = fopen(json_path, "w");
    if (!f) return -1;

    fprintf(f, "{\n");
    fprintf(f, "  \"events\": %d,\n", mem->event_count);
    fprintf(f, "  \"embeddings\": %d,\n", mem->embedding_count);
    fprintf(f, "  \"coherence\": {\n");
    fprintf(f, "    \"entropy\": %.3f,\n", mem->current_coherence.entropy_score);
    fprintf(f, "    \"harmony\": %.3f,\n", mem->current_coherence.harmony_score);
    fprintf(f, "    \"gradient\": %.3f,\n", mem->current_coherence.temporal_gradient);
    fprintf(f, "    \"deviation\": %.3f\n", mem->current_coherence.baseline_deviation);
    fprintf(f, "  },\n");
    fprintf(f, "  \"drift\": %.3f,\n", mem->drift_history[(mem->drift_history_idx - 1) % 100]);
    fprintf(f, "  \"avg_drift_20\": %.3f,\n", qallow_temporal_memory_get_average_drift(mem, 20));
    fprintf(f, "  \"trend\": %d\n", qallow_temporal_memory_get_drift_trend(mem));
    fprintf(f, "}\n");

    fclose(f);
    return 0;
}

void qallow_temporal_memory_print_stats(const qallow_temporal_memory_t* mem) {
    if (!mem) return;

    printf("\n╔════════════════════════════════════════════════════════╗\n");
    printf("║           Temporal Memory Statistics                  ║\n");
    printf("╚════════════════════════════════════════════════════════╝\n");
    printf("Events (L1):              %d (capacity %d)\n",
           mem->event_count, mem->event_capacity);
    printf("Embeddings (L2):          %d (capacity %d)\n",
           mem->embedding_count, mem->embedding_capacity);
    printf("Coherence (L3):\n");
    printf("  Entropy:                %.3f\n", mem->current_coherence.entropy_score);
    printf("  Harmony:                %.3f\n", mem->current_coherence.harmony_score);
    printf("  Temporal Gradient:      %.3f\n", mem->current_coherence.temporal_gradient);
    printf("  Baseline Deviation:     %.3f\n", mem->current_coherence.baseline_deviation);
    printf("Drift Detection:\n");
    printf("  Current Drift:          %.3f\n",
           mem->drift_history[(mem->drift_history_idx - 1) % 100]);
    printf("  Threshold:              %.3f\n", mem->coherence_threshold);
    printf("  Avg Drift (20):         %.3f\n",
           qallow_temporal_memory_get_average_drift(mem, 20));
    printf("  Trend:                  %s\n",
           qallow_temporal_memory_get_drift_trend(mem) > 0 ? "Increasing ⚠️" :
           qallow_temporal_memory_get_drift_trend(mem) < 0 ? "Decreasing ✅" :
           "Stable");
    printf("╚════════════════════════════════════════════════════════╝\n\n");
}
