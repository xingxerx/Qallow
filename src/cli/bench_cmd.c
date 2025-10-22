#include "qallow/module.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

typedef struct {
    const char *name;
    double cpu_time_ms;
    double cuda_time_ms;
    double speedup;
    long iterations;
} benchmark_result_t;

#define MAX_BENCHMARKS 10
static benchmark_result_t results[MAX_BENCHMARKS];
static int result_count = 0;

// Benchmark: CPU vs CUDA comparison
static void benchmark_predict(int iterations) {
    printf("[BENCH] Predict module: %d iterations\n", iterations);
    
    clock_t start = clock();
    for (int i = 0; i < iterations; i++) {
        double x = 0.5 - 0.3;
        double reward = 1.0 / (1.0 + exp(-6.0 * x)) - 0.5;
        (void)reward;
    }
    clock_t end = clock();
    
    double cpu_time = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
    printf("  CPU time: %.3f ms\n", cpu_time);
    
    results[result_count].name = "predict";
    results[result_count].cpu_time_ms = cpu_time;
    results[result_count].iterations = iterations;
    result_count++;
}

// Benchmark: Learning module
static void benchmark_learn(int iterations) {
    printf("[BENCH] Learn module: %d iterations\n", iterations);
    
    clock_t start = clock();
    for (int i = 0; i < iterations; i++) {
        double energy = 0.5;
        double risk = 0.5;
        double reward = 0.2;
        
        double target = 0.25;
        double err = target - reward;
        energy += 0.02 * err;
        risk -= 0.02 * err;
    }
    clock_t end = clock();
    
    double cpu_time = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
    printf("  CPU time: %.3f ms\n", cpu_time);
    
    results[result_count].name = "learn";
    results[result_count].cpu_time_ms = cpu_time;
    results[result_count].iterations = iterations;
    result_count++;
}

// Benchmark: Ethics overhead
static void benchmark_ethics_overhead(int iterations) {
    printf("[BENCH] Ethics overhead: %d iterations\n", iterations);
    
    clock_t start = clock();
    for (int i = 0; i < iterations; i++) {
        double risk = 0.5;
        double reward = 0.2;
        
        if (risk > 0.8) reward -= 0.1;
        if (reward < 0.0) reward = 0.0;
    }
    clock_t end = clock();
    
    double cpu_time = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
    printf("  CPU time: %.3f ms\n", cpu_time);
    
    results[result_count].name = "ethics";
    results[result_count].cpu_time_ms = cpu_time;
    results[result_count].iterations = iterations;
    result_count++;
}

// Benchmark: Memory operations
static void benchmark_memory(int iterations) {
    printf("[BENCH] Memory operations: %d iterations\n", iterations);
    
    float *buffer = malloc(1000 * sizeof(float));
    
    clock_t start = clock();
    for (int i = 0; i < iterations; i++) {
        for (int j = 0; j < 1000; j++) {
            buffer[j] = (float)i * 0.1f + (float)j * 0.01f;
        }
    }
    clock_t end = clock();
    
    double cpu_time = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
    printf("  CPU time: %.3f ms\n", cpu_time);
    
    results[result_count].name = "memory";
    results[result_count].cpu_time_ms = cpu_time;
    results[result_count].iterations = iterations;
    result_count++;
    
    free(buffer);
}

// Benchmark: Full pipeline
static void benchmark_full_pipeline(int steps) {
    printf("[BENCH] Full pipeline: %d steps\n", steps);
    
    size_t n = 0;
    const ql_module *mods = ql_get_mind_modules(&n);
    
    float latent_buf[8] = {0};
    ql_state S = {
        .t = 0.0, .reward = 0.0, .energy = 0.5, .risk = 0.5,
        .latent = latent_buf, .latent_bytes = sizeof(latent_buf)
    };
    
    clock_t start = clock();
    for (int k = 0; k < steps; k++) {
        for (size_t i = 0; i < n; i++) {
            mods[i].fn(&S);
        }
        S.t += 1.0;
    }
    clock_t end = clock();
    
    double cpu_time = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
    printf("  CPU time: %.3f ms\n", cpu_time);
    printf("  Throughput: %.1f steps/sec\n", (steps * 1000.0) / cpu_time);
    
    results[result_count].name = "pipeline";
    results[result_count].cpu_time_ms = cpu_time;
    results[result_count].iterations = steps;
    result_count++;
}

// Print benchmark report
static void print_report() {
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║          QALLOW BENCHMARKING REPORT                        ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");
    
    printf("%-15s %-15s %-15s %-15s\n", "Module", "CPU (ms)", "Iterations", "Throughput");
    printf("%-15s %-15s %-15s %-15s\n", "-------", "-------", "----------", "----------");
    
    for (int i = 0; i < result_count; i++) {
        double throughput = (results[i].iterations * 1000.0) / results[i].cpu_time_ms;
        printf("%-15s %-15.3f %-15ld %-15.1f\n",
               results[i].name,
               results[i].cpu_time_ms,
               results[i].iterations,
               throughput);
    }
    
    printf("\n");
}

int qallow_cmd_bench(int argc, char **argv) {
    (void)argc; (void)argv;
    
    printf("[BENCH] Starting Qallow Benchmarking Suite\n\n");
    
    // Run benchmarks
    benchmark_predict(1000000);
    benchmark_learn(1000000);
    benchmark_ethics_overhead(1000000);
    benchmark_memory(10000);
    benchmark_full_pipeline(100);
    
    // Print report
    print_report();
    
    return 0;
}

