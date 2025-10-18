#include "bend_bridge.h"
#include <stdio.h>
#include <stdlib.h>

static int run_to_file(const char* cmd, const char* out_csv) {
    char full[1024];
    snprintf(full, sizeof(full), "%s > %s", cmd, out_csv);
    return system(full);
}

int bend_phase12_csv(const char* bend_bin, const char* out_csv, int ticks, float eps) {
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "%s run bend/phase12.bend %d %.6f", bend_bin, ticks, eps);
    return run_to_file(cmd, out_csv);
}

int bend_phase13_csv(const char* bend_bin, const char* out_csv, int nodes, int ticks, float k) {
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "%s run bend/phase13.bend %d %d %.6f", bend_bin, nodes, ticks, k);
    return run_to_file(cmd, out_csv);
}
