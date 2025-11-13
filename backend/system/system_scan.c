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
