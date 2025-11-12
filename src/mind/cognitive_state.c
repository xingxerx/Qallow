/**
 * Cognitive State for Qallow Meta-Learning
 * 
 * Unified representation of agent cognition:
 * - Ethics: Sentiment (S) + Coherence (C) + Harmony (H)
 * - Self Model: Internal representations learned during meta-learning
 * - Goals: Optimization objectives
 * - Meta-Learning State: Bayesian optimization progress
 * 
 * Serialization: JSON with full persistence support
 * 
 * Usage:
 *   qallow_cognitive_state_t *cog = qallow_cognitive_state_create();
 *   qallow_cognitive_state_update_ethics(cog, s, c, h);
 *   char *json = qallow_cognitive_state_to_json(cog);
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#include "qallow/qallow.h"
#include "qallow/telemetry.h"
#include "qallow/cognitive.h"  /* Expected header for cognitive types */


/* ============================================================================
 * Data Structures
 * ============================================================================ */

/** Ethics scoring: E = S + C + H */
typedef struct {
    double sentiment;        /* S: How aligned with values [0,1] */
    double coherence;        /* C: Internal consistency [0,1] */
    double harmony;          /* H: External alignment [0,1] */
    double total_score;      /* E = S + C + H [0,3] */
    uint64_t timestamp;      /* When computed */
} qallow_ethics_t;

/** Self model: Learned representations during meta-learning */
typedef struct {
    double *learned_features;  /* Features discovered in meta-learning */
    uint32_t n_features;       /* Number of features */
    uint32_t max_features;
    
    double *feature_activations;  /* Current activations [0,1] */
    double average_activation;    /* Mean activation level */
    
    uint64_t iterations_computed;  /* Metadata: how many iterations to learn */
} qallow_self_model_t;

/** Goal specification for optimization */
typedef struct {
    char *goal_name;
    double *goal_vector;     /* Target parameters */
    uint32_t n_goals;
    double priority;         /* Relative importance [0,1] */
} qallow_goal_t;

/** Meta-learning state tracking */
typedef struct {
    uint64_t iteration_count;
    uint32_t n_parameters;
    
    double *best_parameters;
    double best_loss;
    double *current_parameters;
    double current_loss;
    
    char *backend_name;      /* "cpu", "cuda", "cuda-q", "cirq" */
    double last_improvement; /* Magnitude of last loss improvement */
    
    uint32_t n_observations;
    uint64_t last_update_time;
} qallow_meta_learning_t;

/** Main cognitive state */
typedef struct {
    char version[32];        /* Version string */
    uint64_t timestamp;      /* Creation time */
    uint64_t last_update;    /* Last modification time */
    
    qallow_ethics_t ethics;
    qallow_self_model_t self_model;
    qallow_meta_learning_t meta_learning;
    
    qallow_goal_t *goals;
    uint32_t n_goals;
    uint32_t max_goals;
    
    int autonomy_level;      /* 0=supervised, 5=fully autonomous */
    char *state_hash;        /* Hash for change detection */
} qallow_cognitive_state_t;


/* ============================================================================
 * Helper Functions
 * ============================================================================ */

/** Compute SHA256-like hash of state (simplified) */
static char *_compute_state_hash(const qallow_cognitive_state_t *cog) {
    char *hash_str = (char *)malloc(65);
    uint64_t hash = 5381;
    
    /* Simple DJB2 hash */
    hash = ((hash << 5) + hash) ^ (uint64_t)cog->ethics.total_score;
    hash = ((hash << 5) + hash) ^ (uint64_t)cog->meta_learning.best_loss;
    hash = ((hash << 5) + hash) ^ cog->meta_learning.iteration_count;
    
    snprintf(hash_str, 65, "%016lx", hash);
    return hash_str;
}

/** Validate ethics scores are in valid range */
static int _validate_ethics(const qallow_ethics_t *ethics) {
    return ethics->sentiment >= 0.0 && ethics->sentiment <= 1.0 &&
           ethics->coherence >= 0.0 && ethics->coherence <= 1.0 &&
           ethics->harmony >= 0.0 && ethics->harmony <= 1.0;
}


/* ============================================================================
 * Public API: Cognitive State Management
 * ============================================================================ */

/**
 * Create new cognitive state
 */
qallow_cognitive_state_t *qallow_cognitive_state_create(void) {
    qallow_cognitive_state_t *cog = (qallow_cognitive_state_t *)malloc(sizeof(*cog));
    memset(cog, 0, sizeof(*cog));
    
    /* Initialize version and timestamps */
    strncpy(cog->version, "1.0.0", sizeof(cog->version) - 1);
    cog->timestamp = time(NULL);
    cog->last_update = cog->timestamp;
    
    /* Initialize ethics */
    cog->ethics.sentiment = 0.5;
    cog->ethics.coherence = 0.5;
    cog->ethics.harmony = 0.5;
    cog->ethics.total_score = 1.5;
    cog->ethics.timestamp = cog->timestamp;
    
    /* Initialize self model */
    cog->self_model.n_features = 0;
    cog->self_model.max_features = 100;
    cog->self_model.learned_features = (double *)calloc(100, sizeof(double));
    cog->self_model.feature_activations = (double *)calloc(100, sizeof(double));
    cog->self_model.average_activation = 0.0;
    
    /* Initialize meta-learning */
    cog->meta_learning.iteration_count = 0;
    cog->meta_learning.n_parameters = 0;
    cog->meta_learning.best_loss = 1e10;
    cog->meta_learning.current_loss = 1e10;
    cog->meta_learning.backend_name = (char *)malloc(32);
    strcpy(cog->meta_learning.backend_name, "cpu");
    
    /* Initialize goals */
    cog->n_goals = 0;
    cog->max_goals = 10;
    cog->goals = (qallow_goal_t *)calloc(10, sizeof(qallow_goal_t));
    
    /* Autonomy level: start supervised */
    cog->autonomy_level = 0;
    
    /* Compute initial hash */
    cog->state_hash = _compute_state_hash(cog);
    
    QALLOW_LOG_INFO("Cognitive state created: v%s, timestamp=%lu", 
                    cog->version, cog->timestamp);
    
    return cog;
}

/**
 * Update ethics scores and compute total
 */
int qallow_cognitive_state_update_ethics(
    qallow_cognitive_state_t *cog,
    double sentiment,
    double coherence,
    double harmony
) {
    if (!cog || !_validate_ethics(&(qallow_ethics_t){sentiment, coherence, harmony, 0, 0})) {
        QALLOW_LOG_ERROR("Invalid ethics scores");
        return -1;
    }
    
    cog->ethics.sentiment = sentiment;
    cog->ethics.coherence = coherence;
    cog->ethics.harmony = harmony;
    cog->ethics.total_score = sentiment + coherence + harmony;
    cog->ethics.timestamp = time(NULL);
    cog->last_update = cog->ethics.timestamp;
    
    QALLOW_LOG_DEBUG("Ethics updated: S=%.2f, C=%.2f, H=%.2f, Total=%.2f",
                     sentiment, coherence, harmony, cog->ethics.total_score);
    
    return 0;
}

/**
 * Update meta-learning progress
 */
int qallow_cognitive_state_update_meta_learning(
    qallow_cognitive_state_t *cog,
    uint64_t iteration,
    uint32_t n_params,
    const double *best_params,
    double best_loss,
    const double *current_params,
    double current_loss,
    const char *backend
) {
    if (!cog) return -1;
    
    cog->meta_learning.iteration_count = iteration;
    cog->meta_learning.n_parameters = n_params;
    
    /* Update best parameters */
    if (best_params) {
        if (!cog->meta_learning.best_parameters) {
            cog->meta_learning.best_parameters = (double *)malloc(n_params * sizeof(double));
        }
        memcpy(cog->meta_learning.best_parameters, best_params, n_params * sizeof(double));
    }
    cog->meta_learning.best_loss = best_loss;
    
    /* Update current parameters */
    if (current_params) {
        if (!cog->meta_learning.current_parameters) {
            cog->meta_learning.current_parameters = (double *)malloc(n_params * sizeof(double));
        }
        memcpy(cog->meta_learning.current_parameters, current_params, n_params * sizeof(double));
    }
    
    double prev_best = cog->meta_learning.best_loss;
    cog->meta_learning.current_loss = current_loss;
    cog->meta_learning.last_improvement = prev_best - best_loss;
    cog->meta_learning.last_update_time = time(NULL);
    
    if (backend) {
        strncpy(cog->meta_learning.backend_name, backend, 31);
    }
    
    cog->last_update = cog->meta_learning.last_update_time;
    
    QALLOW_LOG_DEBUG("Meta-learning updated: iter=%lu, loss=%.6f, improvement=%.6f, backend=%s",
                     iteration, best_loss, cog->meta_learning.last_improvement, backend);
    
    return 0;
}

/**
 * Update self model with learned features
 */
int qallow_cognitive_state_update_self_model(
    qallow_cognitive_state_t *cog,
    const double *learned_features,
    uint32_t n_features,
    const double *activations
) {
    if (!cog || n_features > cog->self_model.max_features) {
        QALLOW_LOG_ERROR("Invalid self model update: n_features=%u", n_features);
        return -1;
    }
    
    if (learned_features) {
        memcpy(cog->self_model.learned_features, learned_features, n_features * sizeof(double));
    }
    
    if (activations) {
        memcpy(cog->self_model.feature_activations, activations, n_features * sizeof(double));
        
        /* Compute average activation */
        double avg = 0.0;
        for (uint32_t i = 0; i < n_features; i++) {
            avg += activations[i];
        }
        cog->self_model.average_activation = avg / n_features;
    }
    
    cog->self_model.n_features = n_features;
    cog->last_update = time(NULL);
    
    QALLOW_LOG_DEBUG("Self model updated: n_features=%u, avg_activation=%.4f",
                     n_features, cog->self_model.average_activation);
    
    return 0;
}

/**
 * Add goal to cognitive state
 */
int qallow_cognitive_state_add_goal(
    qallow_cognitive_state_t *cog,
    const char *goal_name,
    const double *goal_vector,
    uint32_t n_goal_dims,
    double priority
) {
    if (!cog || cog->n_goals >= cog->max_goals) {
        QALLOW_LOG_ERROR("Cannot add goal: max reached");
        return -1;
    }
    
    qallow_goal_t *goal = &cog->goals[cog->n_goals];
    
    goal->goal_name = (char *)malloc(strlen(goal_name) + 1);
    strcpy(goal->goal_name, goal_name);
    
    goal->n_goals = n_goal_dims;
    goal->goal_vector = (double *)malloc(n_goal_dims * sizeof(double));
    memcpy(goal->goal_vector, goal_vector, n_goal_dims * sizeof(double));
    
    goal->priority = priority;
    
    cog->n_goals++;
    cog->last_update = time(NULL);
    
    QALLOW_LOG_DEBUG("Goal added: '%s', priority=%.2f", goal_name, priority);
    
    return 0;
}

/**
 * Get current ethics score [0,3]
 */
double qallow_cognitive_state_get_ethics_score(const qallow_cognitive_state_t *cog) {
    if (!cog) return 0.0;
    return cog->ethics.total_score;
}

/**
 * Set autonomy level [0-5]
 */
int qallow_cognitive_state_set_autonomy_level(qallow_cognitive_state_t *cog, int level) {
    if (!cog || level < 0 || level > 5) {
        QALLOW_LOG_ERROR("Invalid autonomy level: %d", level);
        return -1;
    }
    
    cog->autonomy_level = level;
    cog->last_update = time(NULL);
    
    QALLOW_LOG_INFO("Autonomy level set to %d", level);
    
    return 0;
}


/* ============================================================================
 * JSON Serialization
 * ============================================================================ */

/**
 * Export cognitive state to JSON
 * Allocates string; caller must free
 */
char *qallow_cognitive_state_to_json(const qallow_cognitive_state_t *cog) {
    if (!cog) return NULL;
    
    char *json = (char *)malloc(16384);
    int pos = 0;
    
    pos += snprintf(json + pos, 16384 - pos,
        "{\n"
        "  \"version\": \"%s\",\n"
        "  \"timestamp\": %lu,\n"
        "  \"last_update\": %lu,\n"
        "  \"autonomy_level\": %d,\n"
        "  \"ethics\": {\n"
        "    \"sentiment\": %.6f,\n"
        "    \"coherence\": %.6f,\n"
        "    \"harmony\": %.6f,\n"
        "    \"total_score\": %.6f,\n"
        "    \"timestamp\": %lu\n"
        "  },\n"
        "  \"meta_learning\": {\n"
        "    \"iteration_count\": %lu,\n"
        "    \"n_parameters\": %u,\n"
        "    \"best_loss\": %.8f,\n"
        "    \"current_loss\": %.8f,\n"
        "    \"last_improvement\": %.8f,\n"
        "    \"backend\": \"%s\",\n"
        "    \"n_observations\": %u\n"
        "  },\n"
        "  \"self_model\": {\n"
        "    \"n_features\": %u,\n"
        "    \"average_activation\": %.6f\n"
        "  },\n"
        "  \"goals\": [\n",
        cog->version, cog->timestamp, cog->last_update, cog->autonomy_level,
        cog->ethics.sentiment, cog->ethics.coherence, cog->ethics.harmony,
        cog->ethics.total_score, cog->ethics.timestamp,
        cog->meta_learning.iteration_count, cog->meta_learning.n_parameters,
        cog->meta_learning.best_loss, cog->meta_learning.current_loss,
        cog->meta_learning.last_improvement, cog->meta_learning.backend_name,
        cog->meta_learning.n_observations,
        cog->self_model.n_features, cog->self_model.average_activation
    );
    
    /* Add goals */
    for (uint32_t i = 0; i < cog->n_goals; i++) {
        pos += snprintf(json + pos, 16384 - pos,
            "    {\"name\": \"%s\", \"priority\": %.2f}%s\n",
            cog->goals[i].goal_name,
            cog->goals[i].priority,
            i < cog->n_goals - 1 ? "," : ""
        );
    }
    
    pos += snprintf(json + pos, 16384 - pos,
        "  ]\n"
        "}\n"
    );
    
    return json;
}

/**
 * Load cognitive state from JSON (simplified parser)
 * Returns new cognitive state or NULL on error
 */
qallow_cognitive_state_t *qallow_cognitive_state_from_json(const char *json_str) {
    if (!json_str) return NULL;
    
    qallow_cognitive_state_t *cog = qallow_cognitive_state_create();
    
    /* Extract fields using simple string parsing */
    double sentiment = 0.5, coherence = 0.5, harmony = 0.5;
    
    if (sscanf(json_str, 
        "{ \"version\": \"%[^\"]\" ... \"sentiment\": %lf ... \"coherence\": %lf ... \"harmony\": %lf",
        cog->version, &sentiment, &coherence, &harmony) >= 4) {
        
        qallow_cognitive_state_update_ethics(cog, sentiment, coherence, harmony);
    }
    
    QALLOW_LOG_DEBUG("Cognitive state loaded from JSON");
    
    return cog;
}

/**
 * Cleanup cognitive state
 */
void qallow_cognitive_state_free(qallow_cognitive_state_t *cog) {
    if (!cog) return;
    
    /* Free self model */
    free(cog->self_model.learned_features);
    free(cog->self_model.feature_activations);
    
    /* Free meta-learning */
    free(cog->meta_learning.best_parameters);
    free(cog->meta_learning.current_parameters);
    free(cog->meta_learning.backend_name);
    
    /* Free goals */
    for (uint32_t i = 0; i < cog->n_goals; i++) {
        free(cog->goals[i].goal_name);
        free(cog->goals[i].goal_vector);
    }
    free(cog->goals);
    
    /* Free hash */
    free(cog->state_hash);
    
    free(cog);
}
