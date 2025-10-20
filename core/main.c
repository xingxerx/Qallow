#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

#include "ethics/ethics_model.h"
#include "runtime/meta_introspect.h"
#include "runtime/dl_integration.h"
#include "qallow_kernels.h"
static void ensure_logs_dir(void) {
    struct stat st;
    if (stat("logs", &st) != 0) {
        mkdir("logs", 0755);
    }
    if (stat("/var/log/qallow", &st) != 0) {
        if (mkdir("/var/log/qallow", 0755) != 0 && errno != EEXIST) {
            /* best-effort */
        }
    }
}

static void log_csv(const char* tag, const char* path) {
    ensure_logs_dir();
    FILE* f = fopen(path, "a");
    if (!f) {
        return;
    }
    fprintf(f, "%s,mode=GPU,ts=%ld\n", tag, (long)time(NULL));
    fclose(f);
}

static void meta_bootstrap_gpu(void) {
    meta_introspect_apply_environment_defaults();
    if (!meta_introspect_enabled()) {
        meta_introspect_enable(1);
    }
    meta_introspect_configure(NULL, NULL);
    meta_introspect_set_gpu_available(1);
}

static float clamp_unit(float value) {
    if (value < 0.0f) return 0.0f;
    if (value > 1.0f) return 1.0f;
    return value;
}

static void meta_record(const char* phase,
                        const char* module,
                        const char* objective_id,
                        float duration_s,
                        float coherence,
                        float ethics) {
    learn_event_t event = {
        .phase = phase,
        .module = module,
        .objective_id = objective_id,
        .duration_s = duration_s,
        .coherence = coherence,
        .ethics = ethics
    };
    meta_introspect_push(&event);
}

static int bench_gpu(void) {
    meta_bootstrap_gpu();

    float mean = 0.0f;
    float energy = 0.0f;

    clock_t start = clock();
    if (qallow_p12_elasticity_gpu(1 << 20, 0.1f, &mean) != 0) {
        fprintf(stderr, "[BENCH] GPU elasticity phase failed\n");
        return 1;
    }
    float duration12 = (float)(clock() - start) / (float)CLOCKS_PER_SEC;

    start = clock();
    if (qallow_p13_harmonic_gpu(1 << 20, 3.0f, &energy) != 0) {
        fprintf(stderr, "[BENCH] GPU harmonic phase failed\n");
        return 1;
    }
    float duration13 = (float)(clock() - start) / (float)CLOCKS_PER_SEC;

    meta_record("phase12", "elasticity", "phase12.elasticity",
                duration12,
                clamp_unit(mean),
                clamp_unit(mean));
    meta_record("phase13", "harmonic", "phase13.harmonic",
                duration13,
                clamp_unit(energy / (float)(1 << 20)),
                clamp_unit(1.0f - (energy / (float)(1 << 20))));
    meta_introspect_flush();

    if (dl_model_is_loaded()) {
        float input_vector[3] = {
            mean,
            (float)(energy / (float)(1 << 20)),
            duration13
        };
        float dl_out[4] = {0};
        int produced = dl_model_infer(input_vector, 3, dl_out, 4);
        if (produced > 0) {
            printf("[DL] bench inference -> %d value(s), first=%.6f\n", produced, dl_out[0]);
        } else {
            fprintf(stderr, "[DL] bench inference failed: %s\n", dl_model_last_error());
        }
    }

    printf("[BENCH] p12.mean=%.6f p13.energy=%.6f\n", mean, energy);
    log_csv("bench_ok", "/var/log/qallow/telemetry.csv");
    return 0;
}

static int run_phases_gpu(void) {
    meta_bootstrap_gpu();
    ethics_model_t model;
    ethics_model_default(&model);

    float mean = 0.0f;
    float energy = 0.0f;
        clock_t start = clock();
        if (qallow_p12_elasticity_gpu(1 << 20, 0.1f, &mean) != 0) {
        fprintf(stderr, "[PHASE] GPU elasticity phase failed\n");
        return 1;
    }
        float duration12 = (float)(clock() - start) / (float)CLOCKS_PER_SEC;

        start = clock();
        if (qallow_p13_harmonic_gpu(1 << 20, 3.0f, &energy) != 0) {
        fprintf(stderr, "[PHASE] GPU harmonic phase failed\n");
        return 1;
    }
        float duration13 = (float)(clock() - start) / (float)CLOCKS_PER_SEC;

    double samples = (double)(1 << 20);
    double safety_metric = fmaxf(mean, 0.0f);
    double clarity_metric = fmin((double)energy / samples, 1.0);
    double human_metric = 0.95;
    ethics_metrics_t metrics;
    metrics.safety = safety_metric;
    metrics.clarity = clarity_metric;
    metrics.human = human_metric;
    ethics_score_details_t details;
    double score = ethics_score_core(&model, &metrics, &details);
    int pass = ethics_score_pass(&model, &metrics, &details);

    float coherence12 = fminf((float)(metrics.safety + 0.1), 1.0f);
    float coherence13 = (float)metrics.clarity;
        meta_record("phase12", "elasticity", "phase12.elasticity",
              duration12,
              clamp_unit(coherence12),
              clamp_unit((float)metrics.safety));
        meta_record("phase13", "harmonic", "phase13.harmonic",
              duration13,
              clamp_unit(coherence13),
              clamp_unit((float)metrics.human));
        meta_introspect_flush();

    if (dl_model_is_loaded()) {
        float input_vector[4] = {
            mean,
            (float)metrics.clarity,
            (float)metrics.human,
            (float)score
        };
        float dl_out[4] = {0};
        int produced = dl_model_infer(input_vector, 4, dl_out, 4);
        if (produced > 0) {
            printf("[DL] model output (first)=%.6f (produced %d)\n", dl_out[0], produced);
            float adjusted_score = clamp_unit(dl_out[0]);
            meta_record("phase16", "dl_audit", "phase16.dl.score",
                        0.0f,
                        adjusted_score,
                        adjusted_score);
        } else {
            fprintf(stderr, "[DL] inference warning: %s\n", dl_model_last_error());
        }
    }

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
