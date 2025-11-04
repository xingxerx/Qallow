#ifndef QALLOW_TEMPORAL_MEMORY_H
#define QALLOW_TEMPORAL_MEMORY_H

/**
 * @file temporal_memory.h
 * @brief Hierarchical temporal memory with drift detection
 * 
 * Three-level memory architecture for AGI stability:
 * 
 * Level 1: Episodic Buffer
 *   - Short-term session logs and execution traces
 *   - Raw event sequences (e.g., "phase 13 executed, coherence=0.95")
 *   - TTL-based cleanup (keep last 1000 events)
 * 
 * Level 2: Semantic Embeddings
 *   - Task categories, ethical rules, domain concepts
 *   - Vector embeddings (128-dim, normalized)
 *   - Stable across sessions (learned from data)
 * 
 * Level 3: Coherence Checks
 *   - Temporal gradient scoring (drift detection)
 *   - Baseline deviation tracking
 *   - Stability metrics (fidelity, entropy, harmony)
 * 
 * Usage:
 *   qallow_temporal_memory_t mem = qallow_temporal_memory_init(1000);
 *   qallow_temporal_memory_log_event(&mem, "phase13_executed", 0.95);
 *   float drift = qallow_temporal_memory_detect_drift(&mem);
 *   qallow_temporal_memory_free(&mem);
 */

#include <stdint.h>
#include <stddef.h>
#include <time.h>

/* ============================================================================
 * Level 1: Episodic Buffer
 * ============================================================================ */

typedef struct {
    uint64_t event_id;           /* Unique event identifier */
    time_t timestamp;            /* When event occurred */
    char event_type[64];         /* Event category (e.g., "phase_execution", "error") */
    char event_description[256]; /* Human-readable description */
    float event_score;           /* Numeric score associated with event (0-1) */
} qallow_episodic_event_t;

/* ============================================================================
 * Level 2: Semantic Embeddings
 * ============================================================================ */

typedef struct {
    char concept_name[128];      /* Concept label (e.g., "ethics_fairness", "task_scheduling") */
    float embedding[128];        /* 128-dimensional vector (normalized to unit sphere) */
    float confidence;            /* How confident we are in this embedding (0-1) */
    int occurrence_count;        /* How many times seen in data */
} qallow_semantic_embedding_t;

/* ============================================================================
 * Level 3: Coherence Checks
 * ============================================================================ */

typedef struct {
    float temporal_gradient;     /* Rate of change in coherence over time */
    float baseline_deviation;    /* How far current state is from baseline */
    float entropy_score;         /* Disorder in current execution */
    float harmony_score;         /* Consistency across modules */
    time_t last_baseline_update; /* When we last updated the baseline */
} qallow_coherence_metrics_t;

/* ============================================================================
 * Main Temporal Memory Structure
 * ============================================================================ */

typedef struct {
    /* Level 1: Episodic Buffer */
    qallow_episodic_event_t* events;
    int event_count;
    int event_capacity;
    uint64_t next_event_id;
    
    /* Level 2: Semantic Embeddings */
    qallow_semantic_embedding_t* embeddings;
    int embedding_count;
    int embedding_capacity;
    
    /* Level 3: Coherence Metrics */
    qallow_coherence_metrics_t current_coherence;
    qallow_coherence_metrics_t baseline_coherence;
    
    /* Drift Detection */
    float coherence_threshold;   /* Alert if deviation > threshold (default 0.1) */
    float drift_history[100];    /* Last 100 drift measurements */
    int drift_history_idx;
} qallow_temporal_memory_t;

/* ============================================================================
 * API: Initialize and Cleanup
 * ============================================================================ */

/**
 * Initialize temporal memory with capacity
 * 
 * @param max_events - Maximum episodic events to keep in buffer
 * @return Initialized temporal memory structure
 */
qallow_temporal_memory_t qallow_temporal_memory_init(int max_events);

/**
 * Free temporal memory
 */
void qallow_temporal_memory_free(qallow_temporal_memory_t* mem);

/* ============================================================================
 * API: Level 1 - Episodic Events
 * ============================================================================ */

/**
 * Log an event to episodic buffer
 * 
 * @param mem - temporal memory
 * @param event_type - category (e.g., "phase_execution", "error", "decision")
 * @param description - human-readable description
 * @param score - numeric score (0-1)
 * @return Event ID
 */
uint64_t qallow_temporal_memory_log_event(qallow_temporal_memory_t* mem,
                                          const char* event_type,
                                          const char* description,
                                          float score);

/**
 * Query recent events of given type
 * 
 * @param mem - temporal memory
 * @param event_type - filter by type (NULL = all)
 * @param max_results - maximum events to return
 * @return Array of events (caller must free)
 */
qallow_episodic_event_t* qallow_temporal_memory_query_events(
    qallow_temporal_memory_t* mem,
    const char* event_type,
    int max_results);

/**
 * Cleanup old events (keep most recent N)
 * 
 * @param mem - temporal memory
 * @param keep_count - how many recent events to retain
 */
void qallow_temporal_memory_prune_events(qallow_temporal_memory_t* mem,
                                         int keep_count);

/* ============================================================================
 * API: Level 2 - Semantic Embeddings
 * ============================================================================ */

/**
 * Add or update semantic embedding
 * 
 * @param mem - temporal memory
 * @param concept_name - concept label
 * @param embedding - 128-dim float array
 * @param confidence - confidence score (0-1)
 * @return 0 on success, -1 on error
 */
int qallow_temporal_memory_add_embedding(qallow_temporal_memory_t* mem,
                                         const char* concept_name,
                                         const float embedding[128],
                                         float confidence);

/**
 * Retrieve semantic embedding by name
 * 
 * @param mem - temporal memory
 * @param concept_name - concept to look up
 * @return Embedding pointer (NULL if not found)
 */
qallow_semantic_embedding_t* qallow_temporal_memory_get_embedding(
    qallow_temporal_memory_t* mem,
    const char* concept_name);

/**
 * Find nearest embedding to a query vector (cosine similarity)
 * 
 * @param mem - temporal memory
 * @param query_embedding - 128-dim query vector
 * @param max_results - how many nearest neighbors
 * @return Array of embeddings sorted by similarity
 */
qallow_semantic_embedding_t* qallow_temporal_memory_nearest_embeddings(
    qallow_temporal_memory_t* mem,
    const float query_embedding[128],
    int max_results);

/* ============================================================================
 * API: Level 3 - Coherence & Drift Detection
 * ============================================================================ */

/**
 * Update coherence metrics
 * 
 * @param mem - temporal memory
 * @param temporal_gradient - rate of change
 * @param baseline_deviation - deviation from baseline
 * @param entropy_score - disorder metric
 * @param harmony_score - consistency metric
 */
void qallow_temporal_memory_update_coherence(qallow_temporal_memory_t* mem,
                                             float temporal_gradient,
                                             float baseline_deviation,
                                             float entropy_score,
                                             float harmony_score);

/**
 * Detect drift in current execution
 * 
 * Compares current coherence against baseline and returns deviation score.
 * Logs alert if exceeds threshold.
 * 
 * @param mem - temporal memory
 * @return Drift score (0 = on baseline, 1 = maximum deviation)
 */
float qallow_temporal_memory_detect_drift(qallow_temporal_memory_t* mem);

/**
 * Set baseline coherence (usually after stable warm-up period)
 * 
 * @param mem - temporal memory
 */
void qallow_temporal_memory_set_baseline(qallow_temporal_memory_t* mem);

/**
 * Get coherence metrics
 * 
 * @param mem - temporal memory
 * @return Current coherence metrics
 */
qallow_coherence_metrics_t qallow_temporal_memory_get_coherence(
    const qallow_temporal_memory_t* mem);

/**
 * Set drift threshold for alerts
 * 
 * @param mem - temporal memory
 * @param threshold - deviation threshold (default 0.1)
 */
void qallow_temporal_memory_set_drift_threshold(qallow_temporal_memory_t* mem,
                                                float threshold);

/* ============================================================================
 * API: Drift History & Statistics
 * ============================================================================ */

/**
 * Get average drift over last N measurements
 * 
 * @param mem - temporal memory
 * @param window_size - how many recent measurements to average
 * @return Average drift score
 */
float qallow_temporal_memory_get_average_drift(const qallow_temporal_memory_t* mem,
                                               int window_size);

/**
 * Get drift trend (increasing or decreasing?)
 * 
 * @param mem - temporal memory
 * @return Trend: -1 (decreasing), 0 (stable), +1 (increasing)
 */
int qallow_temporal_memory_get_drift_trend(const qallow_temporal_memory_t* mem);

/**
 * Export memory to JSON for analysis
 * 
 * @param mem - temporal memory
 * @param json_path - output file path
 * @return 0 on success, -1 on error
 */
int qallow_temporal_memory_export_json(const qallow_temporal_memory_t* mem,
                                        const char* json_path);

/**
 * Print memory statistics to stdout
 * 
 * @param mem - temporal memory
 */
void qallow_temporal_memory_print_stats(const qallow_temporal_memory_t* mem);

#endif  /* QALLOW_TEMPORAL_MEMORY_H */
