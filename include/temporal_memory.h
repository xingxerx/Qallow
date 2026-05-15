/* include/temporal_memory.h */
#ifndef TEMPORAL_MEMORY_H
#define TEMPORAL_MEMORY_H

#include <stddef.h>
#include <stdint.h>
#include <time.h>

#define TM_VECTOR_DIM 128
#define TM_EPISODIC_SIZE 256
#define TM_SEMANTIC_SIZE 1024

typedef struct {
    float vec[TM_VECTOR_DIM];
    time_t timestamp;
    uint32_t access_count;
    float coherence_score;
    char label[64];
} MemoryVector;

typedef struct {
    MemoryVector *episodes;      // short-term episodic buffer
    size_t episode_head;
    size_t episode_count;
    
    MemoryVector *semantics;     // long-term semantic embeddings
    size_t semantic_count;
    

    float drift_threshold;
    float avg_coherence;
    uint64_t coherence_checks;
    

    float *gradient_history;
    size_t gradient_window;
    size_t gradient_index;
} TemporalMemory;

/* Core API */
int tm_init(TemporalMemory *tm, float drift_threshold);
void tm_free(TemporalMemory *tm);

/* Memory operations */
int tm_store_episodic(TemporalMemory *tm, const float *vec, size_t dim);
int tm_store_semantic(TemporalMemory *tm, const float *vec, size_t dim, const char *label);
int tm_retrieve_similar(TemporalMemory *tm, const float *query, size_t dim, MemoryVector **result);

/* Coherence and drift prevention */
float tm_check_coherence(TemporalMemory *tm, const float *vec, size_t dim);
int tm_audit_drift(TemporalMemory *tm, float *drift_report);
int tm_recalibrate(TemporalMemory *tm);

/* Temporal gradient tracking */
int tm_update_gradient(TemporalMemory *tm, float gradient);
float tm_predict_next(const TemporalMemory *tm);

#endif /* TEMPORAL_MEMORY_H */
