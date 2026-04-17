#!/bin/bash

mkdir -p backend/system backend/optimization backend/safety include

# system_scan.c
cat > backend/system/system_scan.c <<'EOFCODE'
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/sysinfo.h>
#include <cpuid.h>
#include <sched.h>
#include <dirent.h>

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

static int get_cpu_core_count(void) {
    return sysconf(_SC_NPROCESSORS_ONLN);
}

static int get_numa_node_count(void) {
    int numa_count = 1;
    DIR* dir = opendir("/sys/devices/system/node/");
    if (dir) {
        numa_count = 0;
        struct dirent* entry;
        while ((entry = readdir(dir)) != NULL) {
            if (strncmp(entry->d_name, "node", 4) == 0) {
                numa_count++;
            }
        }
        closedir(dir);
    }
    return numa_count > 0 ? numa_count : 1;
}

static float get_total_memory_gb(void) {
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        return (float)(info.totalram * info.mem_unit) / (1024.0 * 1024.0 * 1024.0);
    }
    return 0.0f;
}

static int detect_cuda_support(void) {
    if (system("which nvidia-smi > /dev/null 2>&1") == 0) {
        return 1;
    }
    return 0;
}

int system_scan(system_state_t* state) {
    if (!state) return -1;
    memset(state, 0, sizeof(system_state_t));
    
    state->cpu_cores = get_cpu_core_count();
    state->numa_nodes = get_numa_node_count();
    state->total_ram_gb = get_total_memory_gb();
    state->cuda_enabled = detect_cuda_support();
    
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        state->cpu_family = (eax >> 8) & 0xf;
        state->cpu_model = (eax >> 4) & 0xf;
        state->cpu_stepping = eax & 0xf;
    }
    
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        state->cpu_usage = 1.0f - (float)info.freeram / (float)info.totalram;
        state->memory_pressure = (float)(info.totalram - info.freeram) / (float)info.totalram;
        state->load_average = info.loads[0] / (float)(1 << SI_LOAD_SHIFT);
    }
    
    FILE* fp = fopen("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", "r");
    if (fp) {
        fscanf(fp, "%u", &state->cpu_freq_mhz);
        state->cpu_freq_mhz /= 1000;
        fclose(fp);
    }
    
    return 0;
}

void print_system_info(const system_state_t* state) {
    printf("[SYSTEM] Detected Hardware:\n");
    printf("[SYSTEM] CPU: %d cores, %d NUMA nodes, %d MHz\n", 
           state->cpu_cores, state->numa_nodes, state->cpu_freq_mhz);
    printf("[SYSTEM] Memory: %.1f GB total\n", state->total_ram_gb);
    printf("[SYSTEM] GPU: %s\n", state->cuda_enabled ? "CUDA-capable" : "Not detected");
    printf("[SYSTEM] Current Load: %.2f CPU usage, %.2f memory pressure\n", 
           state->cpu_usage, state->memory_pressure);
    printf("\n");
}

void print_optimizer_dashboard(const system_state_t* state, int tick) {
    printf("\r[TICK %04d] CPU: %.1f%% | RAM: %.1f%% | Load: %.2f | Temp: %d°C", 
           tick,
           state->cpu_usage * 100.0f,
           state->memory_pressure * 100.0f,
           state->load_average,
           state->cpu_temp);
    fflush(stdout);
}

void print_optimization_summary(const system_state_t* state) {
    printf("\n\n══════════════════════════════════════════════════\n");
    printf("OPTIMIZATION SUMMARY\n");
    printf("══════════════════════════════════════════════════\n");
    printf("Final CPU usage:        %.1f%%\n", state->cpu_usage * 100.0f);
    printf("Final memory pressure:  %.1f%%\n", state->memory_pressure * 100.0f);
    printf("Average load:           %.2f\n", state->load_average);
    printf("CPU governor:           %s\n", state->cpu_governor);
    printf("══════════════════════════════════════════════════\n");
}
EOFCODE

# phase12_elasticity.c
cat > backend/optimization/phase12_elasticity.c <<'EOFCODE'
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/resource.h>
#include <dirent.h>
#include <ctype.h>

#define CPUFREQ_PATH "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_governor"

static int get_cpu_core_count(void) {
    return sysconf(_SC_NPROCESSORS_ONLN);
}

static int set_cpu_governor(const char* governor) {
    char cmd[512];
    for (int cpu = 0; cpu < get_cpu_core_count(); ++cpu) {
        snprintf(cmd, sizeof(cmd), "echo %s > " CPUFREQ_PATH " 2>/dev/null", governor, cpu);
        if (system(cmd) != 0) return -1;
    }
    return 0;
}

static int adjust_swappiness(int value) {
    char cmd[128];
    snprintf(cmd, sizeof(cmd), "sysctl -w vm.swappiness=%d 2>/dev/null", value);
    return system(cmd);
}

static void adjust_process_niceness(void) {
    DIR* proc_dir = opendir("/proc");
    if (!proc_dir) return;
    
    struct dirent* entry;
    while ((entry = readdir(proc_dir)) != NULL) {
        int pid = atoi(entry->d_name);
        if (pid <= 0) continue;
        
        char comm_path[256];
        snprintf(comm_path, sizeof(comm_path), "/proc/%d/comm", pid);
        FILE* f = fopen(comm_path, "r");
        if (!f) continue;
        
        char comm[256];
        if (fgets(comm, sizeof(comm), f)) {
            comm[strcspn(comm, "\n")] = 0;
            if (strstr(comm, "chrome") || strstr(comm, "firefox") || 
                strstr(comm, "backup") || strstr(comm, "index")) {
                setpriority(PRIO_PROCESS, pid, 5);
            } else if (strstr(comm, "terminal") || strstr(comm, "vim") ||
                     strstr(comm, "emacs") || strstr(comm, "code")) {
                setpriority(PRIO_PROCESS, pid, -5);
            }
        }
        fclose(f);
    }
    closedir(proc_dir);
}

int run_phase12_elasticity(const char* audit_tag, const char* log_path, int ticks, float eps) {
    static int initialization_done = 0;
    
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
    
    system_state_t state;
    system_scan(&state); // This will be linked from system_scan.o
    
    if (state.cpu_usage > 0.8f) {
        if (set_cpu_governor("performance") == 0) {
            printf("[PHASE12] CPU governor set to performance (load %.1f%%)\n", state.cpu_usage * 100);
        }
    } else if (state.cpu_usage < 0.3f) {
        if (set_cpu_governor("powersafe") == 0) {
            printf("[PHASE12] CPU governor set to powersave (load %.1f%%)\n", state.cpu_usage * 100);
        }
    }
    
    if (state.memory_pressure > 0.7f && !initialization_done) {
        adjust_swappiness(10);
        printf("[PHASE12] Memory pressure high (%.1f%%), adjusting vm.swappiness\n", 
               state.memory_pressure * 100);
        
        if (system("modprobe zswap 2>/dev/null") == 0) {
            system("echo 1 > /sys/module/zswap/parameters/enabled 2>/dev/null");
            printf("[PHASE12] Enabled zswap for memory compression\n");
        }
        initialization_done = 1;
    }
    
    adjust_process_niceness();
    
    printf("[PHASE12] Elasticity tick %d: CPU %.1f%%, RAM %.1f%%, governor: %s\n",
           ticks, state.cpu_usage * 100, state.memory_pressure * 100, state.cpu_governor);
    return 0;
}

// Stub for system_scan function that will be linked
extern int system_scan(void* state);
EOFCODE

# phase13_harmonic.c
cat > backend/optimization/phase13_harmonic.c <<'EOFCODE'
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>
#include <unistd.h>
#include <dirent.h>

#define PROC_STAT_PATH "/proc/stat"

typedef struct {
    int cpu_id;
    float load;
    int process_count;
    int should_receive_migration;
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
                unsigned long user, nice, system, idle, iowait, irq, softirq, steal;
                sscanf(line, "cpu%d %lu %lu %lu %lu %lu %lu %lu %lu",
                       &cpu_id, &user, &nice, &system, &idle, &iowait, &irq, &softirq, &steal);
                
                unsigned long total = user + nice + system + idle + iowait + irq + softirq + steal;
                unsigned long active = user + nice + system + irq + softirq + steal;
                
                loads[cpu_id].cpu_id = cpu_id;
                loads[cpu_id].load = total > 0 ? (float)active / (float)total : 0.0f;
            }
        }
    }
    fclose(f);
}

static void count_processes_per_cpu(cpu_load_t* loads, int num_cpus) {
    DIR* proc_dir = opendir("/proc");
    if (!proc_dir) return;
    
    struct dirent* entry;
    while ((entry = readdir(proc_dir)) != NULL) {
        int pid = atoi(entry->d_name);
        if (pid <= 0) continue;
        
        cpu_set_t mask;
        CPU_ZERO(&mask);
        if (sched_getaffinity(pid, sizeof(mask), &mask) == 0) {
            for (int i = 0; i < num_cpus; ++i) {
                if (CPU_ISSET(i, &mask)) {
                    loads[i].process_count++;
                }
            }
        }
    }
    closedir(proc_dir);
}

static void calculate_ideal_distribution(cpu_load_t* loads, int num_cpus, float* ideal_load) {
    float total_load = 0.0f;
    int active_cpus = 0;
    
    for (int i = 0; i < num_cpus; ++i) {
        total_load += loads[i].load;
        if (loads[i].load > 0.01f) active_cpus++;
    }
    
    *ideal_load = active_cpus > 0 ? total_load / active_cpus : 0.0f;
}

int run_phase13_harmonic(const char* audit_tag, const char* log_path, 
                        int nodes, int ticks, float coupling) {
    if (nodes <= 0) nodes = get_cpu_core_count();
    
    cpu_load_t* cpu_loads = calloc(nodes, sizeof(cpu_load_t));
    if (!cpu_loads) return -1;
    
    get_per_cpu_load(cpu_loads, nodes);
    count_processes_per_cpu(cpu_loads, nodes);
    
    float ideal_load;
    calculate_ideal_distribution(cpu_loads, nodes, &ideal_load);
    
    for (int i = 0; i < nodes; ++i) {
        if (cpu_loads[i].load < ideal_load * (1.0f - coupling)) {
            cpu_loads[i].should_receive_migration = 1;
        }
    }
    
    int migrations_performed = 0;
    for (int i = 0; i < nodes; ++i) {
        if (cpu_loads[i].load > ideal_load * (1.0f + coupling)) {
            DIR* proc_dir = opendir("/proc");
            if (!proc_dir) break;
            
            struct dirent* entry;
            while ((entry = readdir(proc_dir)) != NULL) {
                int pid = atoi(entry->d_name);
                if (pid <= 0) continue;
                
                int target_cpu = -1;
                for (int j = 0; j < nodes; ++j) {
                    if (cpu_loads[j].should_receive_migration) {
                        target_cpu = j;
                        break;
                    }
                }
                
                if (target_cpu >= 0) {
                    cpu_set_t mask;
                    CPU_ZERO(&mask);
                    CPU_SET(target_cpu, &mask);
                    
                    if (sched_setaffinity(pid, sizeof(mask), &mask) == 0) {
                        migrations_performed++;
                        cpu_loads[i].process_count--;
                        cpu_loads[target_cpu].process_count++;
                        if (migrations_performed >= 3) break;
                    }
                }
            }
            closedir(proc_dir);
        }
    }
    
    printf("[PHASE13] Harmonic balance: ideal_load=%.2f, migrations=%d\n", 
           ideal_load, migrations_performed);
    
    if (log_path) {
        FILE* log = fopen(log_path, "a");
        if (log) {
            for (int i = 0; i < nodes; ++i) {
                fprintf(log, "%d,%d,%d,%.3f,%d\n", ticks, i, cpu_loads[i].process_count,
                        cpu_loads[i].load, cpu_loads[i].should_receive_migration);
            }
            fclose(log);
        }
    }
    
    free(cpu_loads);
    return 0;
}
EOFCODE

# Add headers
cat > include/system_scan.h <<'EOFCODE'
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
EOFCODE

# Add CMakeLists.txt section at the end
cat >> CMakeLists.txt <<'EOFCODE'

# System Optimizer Files
set(OPTIMIZER_SOURCES
    interface/optimizer_main.c
    backend/system/system_scan.c
    backend/optimization/phase12_elasticity.c
    backend/optimization/phase13_harmonic.c
    backend/optimization/phase14_coherence.c
    backend/optimization/phase15_convergence.c
)

add_executable(qallow_optimizer ${OPTIMIZER_SOURCES})
target_link_libraries(qallow_optimizer qallow_core m pthread)
target_include_directories(qallow_optimizer PRIVATE ${CMAKE_SOURCE_DIR}/include)
EOFCODE

echo "All optimizer files created successfully!"
