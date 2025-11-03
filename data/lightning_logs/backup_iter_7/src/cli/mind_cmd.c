#include "qallow/module.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int parse_int(const char *k, int def){
  const char *v = getenv(k);
  return v ? atoi(v) : def;
}

int qallow_cmd_mind(int argc, char **argv){
  (void)argc; (void)argv;
  size_t n=0; 
  const ql_module *mods = ql_get_mind_modules(&n);
  // init state
  float latent_buf[8] = {0};
  ql_state S = {
    .t=0.0, .reward=0.0, .energy=0.5, .risk=0.5,
    .latent=latent_buf, .latent_bytes=sizeof(latent_buf)
  };
  int steps = parse_int("QALLOW_MIND_STEPS", 50);
  printf("[MIND] steps=%d modules=%zu\n", steps, n);
  for (int k=0; k<steps; ++k){
    for (size_t i=0;i<n;++i){
      ql_status r = mods[i].fn(&S);
      if (r.code){
        fprintf(stderr,"[MIND][%d] %s error: %s\n", k, mods[i].name, r.msg);
        return 1;
      }
    }
    S.t += 1.0;
    printf("[MIND][%03d] reward=%.3f energy=%.3f risk=%.3f\n", k, S.reward, S.energy, S.risk);
  }
  return 0;
}

