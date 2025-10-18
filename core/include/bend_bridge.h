#ifndef BEND_BRIDGE_H
#define BEND_BRIDGE_H

int bend_phase12_csv(const char* bend_bin, const char* out_csv, int ticks, float eps);
int bend_phase13_csv(const char* bend_bin, const char* out_csv, int nodes, int ticks, float k);

#endif /* BEND_BRIDGE_H */
