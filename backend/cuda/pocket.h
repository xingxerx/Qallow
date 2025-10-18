#pragma once
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int pockets;      // N
  int nodes;        // per pocket
  int steps;        // ticks per pocket
  double jitter;    // small dynamics term
} pocket_cfg_t;

// host API (call from C core)
int pocket_spawn_and_run(const pocket_cfg_t* cfg);
// copies merged means into provided arrays (len=nodes)
int pocket_merge_to_host(double* orbital, double* river, double* mycelial);
int pocket_release();

#ifdef __cplusplus
}
#endif
