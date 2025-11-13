#pragma once

typedef struct {
    int cpu_cores;
    int numa_nodes;
    float total_ram_gb;
    unsigned int cpu_family;
    unsigned int cpu_model;
    unsigned int cpu_stepping;
    unsigned int cpu_freq_mhz;
    float cpu_usage;
    float memory_pressure;
    float load_average;
    int cpu_temp;
    char cpu_governor[32];
    int cuda_enabled;
} system_state_t;

int system_scan(system_state_t* state);
void print_system_info(const system_state_t* state);
void print_optimizer_dashboard(const system_state_t* state, int tick);
void print_optimization_summary(const system_state_t* state);
