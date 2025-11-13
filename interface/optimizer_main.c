#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <time.h>

static volatile int g_running = 1;

void signal_handler(int sig) {
    if (sig == SIGINT || sig == SIGTERM) {
        printf("\n[OPTIMIZER] Shutting down...\n");
        g_running = 0;
    }
}

int main(void) {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    printf("╔════════════════════════════════════════╗\n");
    printf("║ QALLOW SYSTEM OPTIMIZER v1.0          ║\n");
    printf("╚════════════════════════════════════════╝\n\n");
    
    printf("[SYSTEM] Starting optimization daemon...\n");
    printf("[SYSTEM] Press Ctrl+C to stop\n\n");
    
    int tick = 0;
    while (g_running && tick < 60) {
        printf("\r[TICK %04d] Running optimization phases...", tick);
        fflush(stdout);
        
        // Run all phases
        extern int run_phase12_elasticity(const char*, const char*, int, float);
        extern int run_phase13_harmonic(const char*, const char*, int, int, float);
        extern int run_phase14_coherence(const char*, const char*, int, int, double);
        extern int run_phase15_convergence(const char*, const char*, int, double);
        
        run_phase12_elasticity("optimizer", NULL, tick, 0.001f);
        run_phase13_harmonic("optimizer", NULL, 0, tick, 0.002f);
        run_phase14_coherence("optimizer", NULL, 0, tick, 0.981f);
        
        if (run_phase15_convergence("optimizer", NULL, tick, 1e-5f)) {
            printf("\n[SYSTEM] Stable configuration achieved at tick %d\n", tick);
            break;
        }
        
        sleep(1);
        tick++;
    }
    
    printf("\n[SYSTEM] Optimization complete. Ran for %d ticks.\n", tick);
    return 0;
}
