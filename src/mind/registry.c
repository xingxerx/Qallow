#include "qallow/module.h"
#include <math.h>
#include <string.h>


extern ql_status mod_attention(ql_state *S);
extern ql_status mod_cross_attention(ql_state *S);
extern ql_status mod_episodic_memory(ql_state *S);
extern ql_status mod_semantic_memory(ql_state *S);
extern ql_status mod_memory_recall(ql_state *S);
extern ql_status mod_memory_consolidation(ql_state *S);
extern ql_status mod_quantum_predict(ql_state *S);
extern ql_status mod_quantum_optimize(ql_state *S);
extern ql_status mod_hybrid_optimize(ql_state *S);
extern ql_status mod_quantum_entangle(ql_state *S);
extern ql_status mod_federated_learn(ql_state *S);
extern ql_status mod_privacy_preserving_learn(ql_state *S);
extern ql_status mod_gradient_compression(ql_state *S);
extern ql_status mod_async_param_server(ql_state *S);
extern ql_status mod_consensus(ql_state *S);
extern ql_status mod_multi_stakeholder_ethics(ql_state *S);
extern ql_status mod_explainability(ql_state *S);
extern ql_status mod_audit_trail(ql_state *S);
extern ql_status mod_conflict_resolution(ql_state *S);
extern ql_status mod_fairness_monitor(ql_state *S);
extern ql_status mod_multi_objective_opt(ql_state *S);
extern ql_status mod_safety_projection(ql_state *S);
// Recursive thinking modules
extern ql_status mod_store_thinking_output(ql_state *S);
extern ql_status mod_load_thinking_input(ql_state *S);
extern ql_status mod_extract_strategy_patterns(ql_state *S);
extern ql_status mod_generate_updated_strategy(ql_state *S);
extern ql_status mod_recursive_thinking_cycle(ql_state *S);
extern ql_status mod_export_thinking_metrics(ql_state *S);


static ql_status mod_model(ql_state *S){

  S->risk   = fmax(0.0, S->risk * 0.95);
  S->energy = fmin(1.0, fmax(0.0, S->energy));
  ql_status r = {0, "model ok"};
  return r;
}


static ql_status mod_predict(ql_state *S){

  double x = S->energy - S->risk;
  S->reward = 1.0/(1.0 + exp(-6.0*x)) - 0.5; // [-0.5,0.5]
  ql_status r = {0, "predict ok"};
  return r;
}


static ql_status mod_plan(ql_state *S){

  if (S->reward < 0.0) S->energy += 0.05; else S->risk -= 0.05;
  ql_status r = {0, "plan ok"};
  return r;
}


static ql_status mod_learn(ql_state *S){

  double target = 0.25; // desired reward margin
  double err = target - S->reward;
  S->energy += 0.02*err;
  S->risk   -= 0.02*err;
  ql_status r = {0, "learn ok"};
  return r;
}


static ql_status mod_abstract(ql_state *S){

  typedef struct { float e, r; } latent2;
  if (!S->latent || S->latent_bytes < sizeof(latent2)) return (ql_status){1,"latent buffer too small"};
  latent2 *L = (latent2*)S->latent;
  L->e = (float)S->energy;
  L->r = (float)S->risk;
  return (ql_status){0,"abstract ok"};
}


static ql_status mod_emotion(ql_state *S){

  if (S->risk > 0.8) S->reward -= 0.1;
  if (S->energy > 1.0) S->energy = 1.0;
  if (S->energy < 0.0) S->energy = 0.0;
  return (ql_status){0,"regulator ok"};
}


static ql_status mod_language(ql_state *S){

  (void)S;
  return (ql_status){0,"language ok"};
}


static ql_status mod_meta(ql_state *S){

  static double prev = 0.0;
  double drift = fabs(S->reward - prev);
  if (drift > 0.2) { S->risk *= 0.97; } // damp
  prev = S->reward;
  return (ql_status){0,"meta ok"};
}

static const ql_module MODS[] = {
  {"model",           mod_model},
  {"predict",         mod_predict},
  {"plan",            mod_plan},
  {"learn",           mod_learn},
  {"abstract",        mod_abstract},
  {"regulator",       mod_emotion},
  {"language",        mod_language},
  {"meta",            mod_meta},
  {"attention",       mod_attention},
  {"cross_attention", mod_cross_attention},
  {"episodic_mem",    mod_episodic_memory},
  {"semantic_mem",    mod_semantic_memory},
  {"memory_recall",   mod_memory_recall},
  {"consolidation",   mod_memory_consolidation},
  {"q_predict",       mod_quantum_predict},
  {"q_optimize",      mod_quantum_optimize},
  {"hybrid_opt",      mod_hybrid_optimize},
  {"q_entangle",      mod_quantum_entangle},
  {"fed_learn",       mod_federated_learn},
  {"privacy_learn",   mod_privacy_preserving_learn},
  {"grad_compress",   mod_gradient_compression},
  {"async_param",     mod_async_param_server},
  {"consensus",       mod_consensus},
  {"multi_ethics",    mod_multi_stakeholder_ethics},
  {"explainability",  mod_explainability},
  {"audit_trail",     mod_audit_trail},
  {"conflict_res",    mod_conflict_resolution},
  {"fairness",        mod_fairness_monitor},
  {"multi_opt",       mod_multi_objective_opt},
  {"safety_proj",     mod_safety_projection},
  {"rec_think_cycle", mod_recursive_thinking_cycle},
  {"store_thinking",  mod_store_thinking_output},
  {"load_thinking",   mod_load_thinking_input},
  {"extract_patterns",mod_extract_strategy_patterns},
  {"gen_strategy",    mod_generate_updated_strategy},
  {"think_metrics",   mod_export_thinking_metrics},
};

const ql_module *ql_get_mind_modules(size_t *count){
  *count = sizeof(MODS)/sizeof(MODS[0]);
  return MODS;
}

