// Semantic Memory Grid - In-memory implementation
// TODO: Upgrade to LMDB for production persistence

#include "phase7.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

// ============================================================================
// INTERNAL STATE
// ============================================================================

static smg_node_t* node_table = NULL;
static int node_count = 0;
static int node_capacity = 0;

static smg_edge_t* edge_table = NULL;
static int edge_count = 0;
static int edge_capacity = 0;

static smg_skill_t* skill_table = NULL;
static int skill_count = 0;
static int skill_capacity = 0;

static bool smg_is_initialized = false;
static char smg_db_path[256] = {0};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

static uint64_t get_timestamp(void) {
    return (uint64_t)time(NULL);
}

static float cosine_similarity(const float* a, const float* b, int dim) {
    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (int i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    if (norm_a == 0.0 || norm_b == 0.0) return 0.0f;
    return (float)(dot / (sqrt(norm_a) * sqrt(norm_b)));
}

static int allocate_node_id(void) {
    return node_count++;
}

// ============================================================================
// INITIALIZATION
// ============================================================================

int smg_init(const char* db_path) {
    if (smg_is_initialized) {
        return 0;  // Already initialized
    }
    
    // Allocate tables
    node_capacity = 10000;
    node_table = (smg_node_t*)calloc(node_capacity, sizeof(smg_node_t));
    if (!node_table) return -1;
    
    edge_capacity = 50000;
    edge_table = (smg_edge_t*)calloc(edge_capacity, sizeof(smg_edge_t));
    if (!edge_table) {
        free(node_table);
        return -1;
    }
    
    skill_capacity = 1000;
    skill_table = (smg_skill_t*)calloc(skill_capacity, sizeof(smg_skill_t));
    if (!skill_table) {
        free(node_table);
        free(edge_table);
        return -1;
    }
    
    node_count = 0;
    edge_count = 0;
    skill_count = 0;
    
    if (db_path) {
        strncpy(smg_db_path, db_path, sizeof(smg_db_path) - 1);
    }
    
    smg_is_initialized = true;
    return 0;
}

void smg_shutdown(void) {
    if (!smg_is_initialized) return;
    
    free(node_table);
    free(edge_table);
    free(skill_table);
    
    node_table = NULL;
    edge_table = NULL;
    skill_table = NULL;
    
    node_count = edge_count = skill_count = 0;
    smg_is_initialized = false;
}

// ============================================================================
// NODE OPERATIONS
// ============================================================================

int smg_upsert_concept(const char* label, const float* embedding, int dim, smg_node_type_t type) {
    if (!smg_is_initialized || !label || !embedding) return -1;
    if (dim > SMG_EMBEDDING_DIM) return -1;
    if (node_count >= node_capacity) return -1;
    
    // Check if concept already exists
    for (int i = 0; i < node_count; i++) {
        if (node_table[i].is_active && strcmp(node_table[i].label, label) == 0) {
            // Update existing
            memcpy(node_table[i].embedding, embedding, dim * sizeof(float));
            node_table[i].timestamp = get_timestamp();
            node_table[i].confidence = 0.9f;  // Refresh confidence
            return node_table[i].id;
        }
    }
    
    // Create new
    int id = allocate_node_id();
    smg_node_t* node = &node_table[id];
    
    node->id = id;
    strncpy(node->label, label, SMG_MAX_LABEL - 1);
    node->type = type;
    memcpy(node->embedding, embedding, dim * sizeof(float));
    node->timestamp = get_timestamp();
    node->confidence = 1.0f;
    node->is_active = true;
    
    return id;
}

int smg_get_node(int node_id, smg_node_t* out) {
    if (!smg_is_initialized || !out) return -1;
    if (node_id < 0 || node_id >= node_count) return -1;
    if (!node_table[node_id].is_active) return -1;
    
    memcpy(out, &node_table[node_id], sizeof(smg_node_t));
    return 0;
}

int smg_delete_node(int node_id) {
    if (!smg_is_initialized) return -1;
    if (node_id < 0 || node_id >= node_count) return -1;
    
    node_table[node_id].is_active = false;
    return 0;
}

// ============================================================================
// EDGE OPERATIONS
// ============================================================================

int smg_link(int src_id, int dst_id, const char* relation, double weight) {
    if (!smg_is_initialized || !relation) return -1;
    if (src_id < 0 || src_id >= node_count) return -1;
    if (dst_id < 0 || dst_id >= node_count) return -1;
    if (edge_count >= edge_capacity) return -1;
    
    // Check if edge already exists
    for (int i = 0; i < edge_count; i++) {
        if (edge_table[i].src_id == src_id && 
            edge_table[i].dst_id == dst_id &&
            strcmp(edge_table[i].relation, relation) == 0) {
            // Update weight
            edge_table[i].weight = weight;
            edge_table[i].timestamp = get_timestamp();
            return 0;
        }
    }
    
    // Create new edge
    smg_edge_t* edge = &edge_table[edge_count++];
    edge->src_id = src_id;
    edge->dst_id = dst_id;
    strncpy(edge->relation, relation, SMG_MAX_REL - 1);
    edge->weight = weight;
    edge->timestamp = get_timestamp();
    edge->confidence = 1.0f;
    
    return 0;
}

int smg_get_edges(int node_id, smg_edge_t* out_edges, int max_edges, int* count) {
    if (!smg_is_initialized || !out_edges || !count) return -1;
    if (node_id < 0 || node_id >= node_count) return -1;
    
    *count = 0;
    for (int i = 0; i < edge_count && *count < max_edges; i++) {
        if (edge_table[i].src_id == node_id || edge_table[i].dst_id == node_id) {
            memcpy(&out_edges[*count], &edge_table[i], sizeof(smg_edge_t));
            (*count)++;
        }
    }
    
    return 0;
}

int smg_delete_edge(int src_id, int dst_id, const char* relation) {
    if (!smg_is_initialized || !relation) return -1;
    
    for (int i = 0; i < edge_count; i++) {
        if (edge_table[i].src_id == src_id && 
            edge_table[i].dst_id == dst_id &&
            strcmp(edge_table[i].relation, relation) == 0) {
            // Shift remaining edges
            memmove(&edge_table[i], &edge_table[i+1], 
                   (edge_count - i - 1) * sizeof(smg_edge_t));
            edge_count--;
            return 0;
        }
    }
    
    return -1;
}

// ============================================================================
// RETRIEVAL (VECTOR SIMILARITY)
// ============================================================================

typedef struct {
    int id;
    float score;
} search_result_t;

static int compare_search_results(const void* a, const void* b) {
    float diff = ((search_result_t*)b)->score - ((search_result_t*)a)->score;
    if (diff > 0.0f) return 1;
    if (diff < 0.0f) return -1;
    return 0;
}

int smg_retrieve(const float* query_vec, int dim, int k, int* out_ids, float* out_scores) {
    if (!smg_is_initialized || !query_vec || !out_ids) return -1;
    if (dim > SMG_EMBEDDING_DIM || k <= 0) return -1;
    
    // Compute similarities for all active nodes
    search_result_t* results = (search_result_t*)malloc(node_count * sizeof(search_result_t));
    if (!results) return -1;
    
    int result_count = 0;
    for (int i = 0; i < node_count; i++) {
        if (!node_table[i].is_active) continue;
        
        float sim = cosine_similarity(query_vec, node_table[i].embedding, dim);
        results[result_count].id = node_table[i].id;
        results[result_count].score = sim;
        result_count++;
    }
    
    // Sort by score descending
    qsort(results, result_count, sizeof(search_result_t), compare_search_results);
    
    // Return top k
    int return_count = (k < result_count) ? k : result_count;
    for (int i = 0; i < return_count; i++) {
        out_ids[i] = results[i].id;
        if (out_scores) out_scores[i] = results[i].score;
    }
    
    free(results);
    return return_count;
}

int smg_retrieve_by_type(const float* query_vec, int dim, smg_node_type_t type, int k, int* out_ids) {
    if (!smg_is_initialized || !query_vec || !out_ids) return -1;
    if (dim > SMG_EMBEDDING_DIM || k <= 0) return -1;
    
    // Filtered retrieval
    search_result_t* results = (search_result_t*)malloc(node_count * sizeof(search_result_t));
    if (!results) return -1;
    
    int result_count = 0;
    for (int i = 0; i < node_count; i++) {
        if (!node_table[i].is_active || node_table[i].type != type) continue;
        
        float sim = cosine_similarity(query_vec, node_table[i].embedding, dim);
        results[result_count].id = node_table[i].id;
        results[result_count].score = sim;
        result_count++;
    }
    
    qsort(results, result_count, sizeof(search_result_t), compare_search_results);
    
    int return_count = (k < result_count) ? k : result_count;
    for (int i = 0; i < return_count; i++) {
        out_ids[i] = results[i].id;
    }
    
    free(results);
    return return_count;
}

// ============================================================================
// SKILL MANAGEMENT
// ============================================================================

int smg_register_skill(const char* name, const char* io_sig, void* code_ptr, smg_skill_t* out) {
    if (!smg_is_initialized || !name || !io_sig) return -1;
    if (skill_count >= skill_capacity) return -1;
    
    smg_skill_t* skill = &skill_table[skill_count++];
    skill->id = skill_count - 1;
    strncpy(skill->name, name, SMG_MAX_LABEL - 1);
    strncpy(skill->io_signature, io_sig, sizeof(skill->io_signature) - 1);
    skill->code_ptr = code_ptr;
    skill->success_score = 0.5;  // Neutral start
    skill->usage_count = 0;
    skill->created_ts = get_timestamp();
    
    if (out) {
        memcpy(out, skill, sizeof(smg_skill_t));
    }
    
    return skill->id;
}

int smg_get_skill(int skill_id, smg_skill_t* out) {
    if (!smg_is_initialized || !out) return -1;
    if (skill_id < 0 || skill_id >= skill_count) return -1;
    
    memcpy(out, &skill_table[skill_id], sizeof(smg_skill_t));
    return 0;
}

int smg_update_skill_score(int skill_id, double score) {
    if (!smg_is_initialized) return -1;
    if (skill_id < 0 || skill_id >= skill_count) return -1;
    
    skill_table[skill_id].success_score = score;
    skill_table[skill_id].usage_count++;
    return 0;
}

// ============================================================================
// MAINTENANCE
// ============================================================================

int smg_compact(double min_weight_threshold, uint64_t max_age_seconds) {
    if (!smg_is_initialized) return -1;
    
    uint64_t now = get_timestamp();
    int removed = 0;
    
    // Remove low-weight or old edges
    for (int i = edge_count - 1; i >= 0; i--) {
        bool should_remove = false;
        
        if (edge_table[i].weight < min_weight_threshold) should_remove = true;
        if ((now - edge_table[i].timestamp) > max_age_seconds) should_remove = true;
        
        if (should_remove) {
            memmove(&edge_table[i], &edge_table[i+1], 
                   (edge_count - i - 1) * sizeof(smg_edge_t));
            edge_count--;
            removed++;
        }
    }
    
    return removed;
}

int smg_decay_edges(double decay_rate) {
    if (!smg_is_initialized) return -1;
    if (decay_rate < 0.0 || decay_rate > 1.0) return -1;
    
    for (int i = 0; i < edge_count; i++) {
        edge_table[i].weight *= (1.0 - decay_rate);
        edge_table[i].confidence *= (1.0 - decay_rate * 0.5);  // Slower decay
    }
    
    return 0;
}

int smg_checkpoint(const char* snapshot_path) {
    if (!smg_is_initialized || !snapshot_path) return -1;
    
    FILE* f = fopen(snapshot_path, "wb");
    if (!f) return -1;
    
    // Write header
    fwrite(&node_count, sizeof(int), 1, f);
    fwrite(&edge_count, sizeof(int), 1, f);
    fwrite(&skill_count, sizeof(int), 1, f);
    
    // Write tables
    fwrite(node_table, sizeof(smg_node_t), node_count, f);
    fwrite(edge_table, sizeof(smg_edge_t), edge_count, f);
    fwrite(skill_table, sizeof(smg_skill_t), skill_count, f);
    
    fclose(f);
    return 0;
}

int smg_verify_integrity(void) {
    if (!smg_is_initialized) return -1;
    
    // Check node references in edges
    for (int i = 0; i < edge_count; i++) {
        if (edge_table[i].src_id >= node_count || edge_table[i].dst_id >= node_count) {
            return -1;  // Invalid reference
        }
    }
    
    return 0;
}
