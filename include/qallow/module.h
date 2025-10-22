#pragma once
#include <stddef.h>

typedef struct {
  double t;           // logical time
  double reward;      // last step utility
  double energy;      // resource estimate
  double risk;        // risk estimate
  void  *latent;      // module-shared pointer (concepts)
  size_t latent_bytes;
} ql_state;

typedef struct {
  int   code;         // 0 ok
  char  msg[160];
} ql_status;

typedef ql_status (*ql_mod_fn)(ql_state *S);

typedef struct {
  const char *name;
  ql_mod_fn   fn;
} ql_module;

// registry
const ql_module *ql_get_mind_modules(size_t *count);

