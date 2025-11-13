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
