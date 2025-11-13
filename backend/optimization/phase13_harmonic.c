#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>
#include <unistd.h>
#include <dirent.h>
#include <ctype.h>

#define PROC_STAT_PATH "/proc/stat"

typedef struct {
    int cpu_id;
    float load;
    int process_count;
} cpu_load_t;

static int get_cpu_core_count(void) {
    return sysconf(_SC_NPROCESSORS_ONLN);
}

static void get_per_cpu_load(cpu_load_t* loads, int num_cpus) {
    FILE* f = fopen(PROC_STAT_PATH, "r");
    if (!f) return;
    
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "cpu", 3) == 0 && isdigit(line[3])) {
            int cpu_id = atoi(line + 3);
            if (cpu_id >= 0 && cpu_id < num_cpus) {
                unsigned long user, nice, system, idle;
                sscanf(line, "cpu%d %lu %lu %lu %lu", &cpu_id, &user, &nice, &system, &idle);
                loads[cpu_id].cpu_id = cpu_id;
                loads[cpu_id].load = (float)(user + nice + system) / (float)(user + nice + system + idle);
            }
        }
    }
    fclose(f);
}

int run_phase13_harmonic(const char* audit_tag, const char* log_path, 
                        int nodes, int ticks, float coupling) {
    if (nodes <= 0) nodes = get_cpu_core_count();
    cpu_load_t* cpu_loads = calloc(nodes, sizeof(cpu_load_t));
    if (!cpu_loads) return -1;
    
    get_per_cpu_load(cpu_loads, nodes);
    
    float avg_load = 0;
    for (int i = 0; i < nodes; ++i) avg_load += cpu_loads[i].load;
    avg_load /= nodes;
    
    printf("[PHASE13] Load balance: avg=%.2f\n", avg_load);
    free(cpu_loads);
    return 0;
}
