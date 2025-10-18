#ifndef PHASE7_H
#define PHASE7_H

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// PHASE 7: PROACTIVE AGI LAYER
// ============================================================================
// Modules: Semantic Memory Grid, Goal Synthesizer, Transfer Engine, Self-Reflection
// Safety: E=S+C+H gate on all commits and external actions

// ============================================================================
// SEMANTIC MEMORY GRID (SMG)
// ============================================================================

#define SMG_MAX_LABEL 128
#define SMG_MAX_REL 32
#define SMG_EMBEDDING_DIM 768
#define SMG_MAX_CONCEPTS 100000
#define SMG_MAX_EDGES 500000

typedef enum {
    SMG_NODE_CONCEPT,
    SMG_NODE_SKILL,
    SMG_NODE_GOAL,
    SMG_NODE_EVENT,
    SMG_NODE_ENTITY
} smg_node_type_t;

typedef struct {
    int id;
    char label[SMG_MAX_LABEL];
    smg_node_type_t type;
    float embedding[SMG_EMBEDDING_DIM];
    uint64_t timestamp;
    float confidence;
    bool is_active;
} smg_node_t;

typedef struct {
    int src_id;
    int dst_id;
    char relation[SMG_MAX_REL];
    double weight;
    uint64_t timestamp;
    float confidence;
} smg_edge_t;

typedef struct {
    int id;
    char name[SMG_MAX_LABEL];
    char io_signature[256];
    void* code_ptr;  // Function pointer or skill reference
    double success_score;
    int usage_count;
    uint64_t created_ts;
} smg_skill_t;

// SMG initialization and lifecycle
int smg_init(const char* db_path);
void smg_shutdown(void);
int smg_checkpoint(const char* snapshot_path);
int smg_verify_integrity(void);

// Node operations
int smg_upsert_concept(const char* label, const float* embedding, int dim, smg_node_type_t type);
int smg_get_node(int node_id, smg_node_t* out);
int smg_delete_node(int node_id);

// Edge operations
int smg_link(int src_id, int dst_id, const char* relation, double weight);
int smg_get_edges(int node_id, smg_edge_t* out_edges, int max_edges, int* count);
int smg_delete_edge(int src_id, int dst_id, const char* relation);

// Retrieval (vector similarity)
int smg_retrieve(const float* query_vec, int dim, int k, int* out_ids, float* out_scores);
int smg_retrieve_by_type(const float* query_vec, int dim, smg_node_type_t type, int k, int* out_ids);

// Skill management
int smg_register_skill(const char* name, const char* io_sig, void* code_ptr, smg_skill_t* out);
int smg_get_skill(int skill_id, smg_skill_t* out);
int smg_update_skill_score(int skill_id, double score);

// Compaction and maintenance
int smg_compact(double min_weight_threshold, uint64_t max_age_seconds);
int smg_decay_edges(double decay_rate);

// ============================================================================
// GOAL SYNTHESIZER (GS)
// ============================================================================

#define GS_MAX_GOALS 256
#define GS_MAX_DESC 256
#define GS_MAX_CONSTRAINTS 16

typedef enum {
    GOAL_STATUS_PROPOSED,
    GOAL_STATUS_COMMITTED,
    GOAL_STATUS_ACTIVE,
    GOAL_STATUS_COMPLETED,
    GOAL_STATUS_FAILED,
    GOAL_STATUS_REJECTED
} goal_status_t;

typedef struct {
    char id[32];
    char description[GS_MAX_DESC];
    double priority;
    double risk;
    double clarity;
    double cost;
    double benefit;  // H score
    goal_status_t status;
    uint64_t created_ts;
    uint64_t committed_ts;
    char constraints[GS_MAX_CONSTRAINTS][128];
    int constraint_count;
    int smg_node_id;  // Link to SMG
} goal_t;

typedef struct {
    goal_t goals[GS_MAX_GOALS];
    int goal_count;
    double w_benefit;
    double w_risk;
    double w_clarity;
    double w_cost;
    uint64_t last_synthesis_ts;
} goal_synthesizer_t;

// GS initialization
int gs_init(goal_synthesizer_t* gs);
void gs_shutdown(goal_synthesizer_t* gs);

// Goal generation
int gs_propose(goal_synthesizer_t* gs, const char* input_text, goal_t* out_goals, int max_goals);
int gs_propose_from_telemetry(goal_synthesizer_t* gs, const void* telemetry_data, goal_t* out_goals, int max_goals);

// Goal scoring: priority = w1*H - w2*Risk + w3*Clarity - w4*Cost
double gs_score_goal(const goal_synthesizer_t* gs, const goal_t* goal);
int gs_rank_goals(goal_synthesizer_t* gs);

// Goal commitment (passes through ethics gate)
int gs_commit(goal_synthesizer_t* gs, const char* goal_id, const void* ethics_state);
int gs_reject(goal_synthesizer_t* gs, const char* goal_id, const char* reason);

// Goal lifecycle
int gs_activate(goal_synthesizer_t* gs, const char* goal_id);
int gs_complete(goal_synthesizer_t* gs, const char* goal_id, double outcome_score);
int gs_fail(goal_synthesizer_t* gs, const char* goal_id, const char* reason);

// Query
int gs_get_goal(const goal_synthesizer_t* gs, const char* goal_id, goal_t* out);
int gs_list_goals(const goal_synthesizer_t* gs, goal_status_t status, goal_t* out_goals, int max_goals);

// ============================================================================
// TRANSFER ENGINE (TE)
// ============================================================================

#define TE_MAX_STEPS 64
#define TE_MAX_PLAN_VARIANTS 16
#define TE_MAX_DOMAIN_SIG 128

typedef struct {
    char action[128];
    int required_skill_id;
    int pocket_id;  // Which pocket to simulate in
    double expected_utility;
    char constraints[256];
} plan_step_t;

typedef struct {
    char plan_id[32];
    char goal_id[32];
    plan_step_t steps[TE_MAX_STEPS];
    int step_count;
    double expected_success_prob;
    double expected_benefit;
    double risk_cost;
    double compute_cost;
    double expected_utility;  // EU = SuccessProb * Benefit - RiskCost - ComputeCost
    uint64_t created_ts;
} plan_t;

typedef struct {
    plan_t plans[TE_MAX_PLAN_VARIANTS];
    int plan_count;
    char domain_cache[64][TE_MAX_DOMAIN_SIG];
    int domain_count;
} transfer_engine_t;

// TE initialization
int te_init(transfer_engine_t* te);
void te_shutdown(transfer_engine_t* te);

// Planning
int te_plan(transfer_engine_t* te, const char* goal_id, const goal_t* goal, plan_t* out_plans, int max_plans);
int te_select_best_plan(const transfer_engine_t* te, const plan_t* plans, int plan_count);

// Skill adaptation
int te_adapt(transfer_engine_t* te, int skill_id, const char* domain_sig, int* out_adapted_skill_id);
int te_cache_domain(transfer_engine_t* te, const char* domain_sig);

// Execution helpers
double te_compute_expected_utility(const plan_t* plan);
int te_assign_pockets(plan_t* plan, int num_pockets);

// ============================================================================
// SELF-REFLECTION CORE (SRC)
// ============================================================================

#define SRC_MAX_NOTES 512
#define SRC_MAX_FLAWS 32

typedef struct {
    char run_id[32];
    char plan_id[32];
    char goal_id[32];
    char flaw_description[256];
    char suggested_fix[256];
    double severity;  // 0.0 = minor, 1.0 = critical
    uint64_t detected_ts;
} reflection_flaw_t;

typedef struct {
    char run_id[32];
    double confidence;
    double drift;
    double outcome_score;
    reflection_flaw_t flaws[SRC_MAX_FLAWS];
    int flaw_count;
    char notes[SRC_MAX_NOTES];
    bool needs_resimulation;
    uint64_t review_ts;
} reflection_result_t;

typedef struct {
    reflection_result_t results[256];
    int result_count;
    double drift_threshold;
    double resimulation_threshold;
} self_reflection_t;

// SRC initialization
int src_init(self_reflection_t* src);
void src_shutdown(self_reflection_t* src);

// Reflection operations
int src_review(self_reflection_t* src, const char* run_id, const plan_t* plan, const void* outcome, reflection_result_t* out);
double src_score(const self_reflection_t* src, const char* run_id, double* confidence_out, double* drift_out);

// Analysis
int src_detect_flaws(const plan_t* plan, const void* outcome, reflection_flaw_t* out_flaws, int max_flaws);
bool src_should_resimulate(const self_reflection_t* src, const reflection_result_t* result);

// Learning
int src_update_smg(const reflection_result_t* result);
int src_improve_plan(const plan_t* original, const reflection_result_t* reflection, plan_t* improved);

// ============================================================================
// PHASE 7 UNIFIED STATE
// ============================================================================

typedef struct {
    goal_synthesizer_t gs;
    transfer_engine_t te;
    self_reflection_t src;
    bool smg_initialized;
    bool phase7_active;
    uint64_t session_start_ts;
    FILE* telemetry_phase7;
    char snapshot_dir[256];
} phase7_state_t;

// Phase 7 lifecycle
int phase7_init(phase7_state_t* state, const char* data_dir);
void phase7_shutdown(phase7_state_t* state);
int phase7_checkpoint(phase7_state_t* state);

// Main execution loop integration
int phase7_tick(phase7_state_t* state, const void* telemetry_data, const void* ethics_state);

// Telemetry
int phase7_log_goal(phase7_state_t* state, const goal_t* goal);
int phase7_log_plan(phase7_state_t* state, const plan_t* plan);
int phase7_log_reflection(phase7_state_t* state, const reflection_result_t* result);

// Governance integration
int phase7_audit(const phase7_state_t* state);
bool phase7_check_hard_stops(const phase7_state_t* state, const void* ethics_state);

#ifdef __cplusplus
}
#endif

#endif // PHASE7_H
