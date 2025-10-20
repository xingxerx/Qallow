#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "qallow_kernels.h"
#include "ethics/ethics_model.h"

static void log_csv(const char* tag, const char* path) {
    FILE* f = fopen(path, "a");
    if (!f) {
        return;
    }
    fprintf(f, "%s,mode=GPU,ts=%ld\n", tag, (long)time(NULL));
    fclose(f);
}

static int bench_gpu(void) {
    float mean = 0.0f;
    float energy = 0.0f;
    if (qallow_p12_elasticity_gpu(1 << 20, 0.1f, &mean) != 0) {
        fprintf(stderr, "[BENCH] GPU elasticity phase failed\n");
        return 1;
    }
    if (qallow_p13_harmonic_gpu(1 << 20, 3.0f, &energy) != 0) {
        fprintf(stderr, "[BENCH] GPU harmonic phase failed\n");
        return 1;
    }
    printf("[BENCH] p12.mean=%.6f p13.energy=%.6f\n", mean, energy);
    log_csv("bench_ok", "/var/log/qallow/telemetry.csv");
    return 0;
}

static int run_phases_gpu(void) {
    ethics_model_t model;
    ethics_model_default(&model);

    float mean = 0.0f;
    float energy = 0.0f;
    if (qallow_p12_elasticity_gpu(1 << 20, 0.1f, &mean) != 0) {
        fprintf(stderr, "[PHASE] GPU elasticity phase failed\n");
        return 1;
    }
    if (qallow_p13_harmonic_gpu(1 << 20, 3.0f, &energy) != 0) {
        fprintf(stderr, "[PHASE] GPU harmonic phase failed\n");
        return 1;
    }
    double samples = (double)(1 << 20);
    ethics_metrics_t metrics;
    metrics.safety = mean;
    metrics.clarity = energy / samples;
    metrics.human = 0.95;
    ethics_score_details_t details;
    double score = ethics_score_core(&model, &metrics, &details);
    int pass = ethics_score_pass(&model, &metrics, &details);

    printf("[PHASE] p12.elasticity.mean=%.6f  p13.harmonic.energy=%.6f  ethics=%.2f  pass=%d\n",
           mean, energy, score, pass);
    log_csv("phase_run_ok", "/var/log/qallow/telemetry.csv");
    return pass ? 0 : 1;
}

int main(int argc, char** argv) {
    int accelerator = 0;
    int bench = 0;

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (strcmp(arg, "run") == 0) {
            continue;
        }
        if (strcmp(arg, "--accelerator") == 0) {
            accelerator = 1;
            continue;
        }
        if (strcmp(arg, "--bench") == 0) {
            bench = 1;
            continue;
        }
        if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            printf("Usage: %s [run] [--accelerator] [--bench]\n", argv[0]);
            return 0;
        }
    }

    if (bench) {
        return bench_gpu();
    }

    if (accelerator) {
        return run_phases_gpu();
    }

    printf("Qallow Unified CUDA entry point active.\n");
    return 0;
}
