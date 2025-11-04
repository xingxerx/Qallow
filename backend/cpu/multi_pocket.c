#include "multi_pocket.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#if CUDA_ENABLED
#include <cuda_runtime.h>
extern int multi_pocket_cuda_run(const pocket_params_t* params,
                                 const qallow_state_t* initial_state,
                                 int num_pockets,
                                 int num_ticks,
                                 int node_count,
                                 float* host_values_out,
                                 float* avg_coherence_out,
                                 float* avg_deco_out,
                                 float* final_coherence_out,
                                 double* total_elapsed_ms);
#endif





void multi_pocket_init(multi_pocket_scheduler_t* scheduler, int num_pockets) {
    if (!scheduler) return;

    memset(scheduler, 0, sizeof(multi_pocket_scheduler_t));

    scheduler->num_pockets = (num_pockets > MAX_POCKETS) ? MAX_POCKETS : num_pockets;


#if CUDA_ENABLED
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    if (device_count > 0) {
        for (int i = 0; i < scheduler->num_pockets; i++) {
            cudaStreamCreate(&scheduler->streams[i]);
        }
        scheduler->streams_initialized = true;
        printf("[MULTI-POCKET] Initialized %d CUDA streams\n", scheduler->num_pockets);
    } else {
        scheduler->streams_initialized = false;
        printf("[MULTI-POCKET] CUDA not available, using CPU execution\n");
    }
#else
    printf("[MULTI-POCKET] CPU-only build\n");
#endif


    sprintf(scheduler->master_telemetry_file, "qallow_multi_pocket.csv");
    scheduler->master_telemetry = fopen(scheduler->master_telemetry_file, "w");

    if (scheduler->master_telemetry) {
        fprintf(scheduler->master_telemetry,
                "tick,pocket_id,coherence,decoherence,ethics,confidence,runtime_ms\n");
    }

    printf("[MULTI-POCKET] Scheduler initialized with %d pockets\n", scheduler->num_pockets);
}


void multi_pocket_cleanup(multi_pocket_scheduler_t* scheduler) {
    if (!scheduler) return;

#if CUDA_ENABLED
    if (scheduler->streams_initialized) {
        for (int i = 0; i < scheduler->num_pockets; i++) {
            cudaStreamDestroy(scheduler->streams[i]);
        }
    }
#endif

    if (scheduler->master_telemetry) {
        fclose(scheduler->master_telemetry);
    }

    printf("[MULTI-POCKET] Scheduler cleaned up\n");
}


void multi_pocket_set_params(multi_pocket_scheduler_t* scheduler,
                             int pocket_id,
                             const pocket_params_t* params) {
    if (!scheduler || pocket_id >= scheduler->num_pockets || !params) return;

    memcpy(&scheduler->params[pocket_id], params, sizeof(pocket_params_t));
    scheduler->params[pocket_id].pocket_id = pocket_id;


    sprintf(scheduler->params[pocket_id].telemetry_file,
            "pocket_%d.csv", pocket_id);
}


void multi_pocket_generate_random_params(multi_pocket_scheduler_t* scheduler) {
    if (!scheduler) return;

    for (int i = 0; i < scheduler->num_pockets; i++) {
        pocket_params_t params;
        params.pocket_id = i;
        params.learning_rate = 0.001f + (float)rand() / RAND_MAX * 0.009f;  // 0.001-0.01
        params.noise_level = 0.01f + (float)rand() / RAND_MAX * 0.04f;      // 0.01-0.05
        params.stability_bias = 0.9f + (float)rand() / RAND_MAX * 0.1f;     // 0.9-1.0
        params.thread_count = 2 + rand() % 7;                                // 2-8 threads

        multi_pocket_set_params(scheduler, i, &params);

        printf("[POCKET %d] LR=%.4f Noise=%.3f Bias=%.2f Threads=%d\n",
               i, params.learning_rate, params.noise_level,
               params.stability_bias, params.thread_count);
    }
}


static void execute_single_pocket_cpu(const pocket_params_t* params,
                                     const qallow_state_t* initial_state,
                                     int num_ticks,
                                     pocket_result_t* result) {
    clock_t start = clock();


    qallow_state_t pocket_state;
    memcpy(&pocket_state, initial_state, sizeof(qallow_state_t));


    FILE* telemetry = fopen(params->telemetry_file, "w");
    if (telemetry) {
        fprintf(telemetry, "tick,coherence,decoherence,orbital,river,mycelial\n");
    }


    float total_coherence = 0.0f;
    float total_decoherence = 0.0f;


    for (int tick = 0; tick < num_ticks; tick++) {
        qallow_kernel_tick(&pocket_state);


        for (int i = 0; i < NUM_OVERLAYS; i++) {
            for (int j = 0; j < pocket_state.overlays[i].node_count; j++) {
                float noise = ((float)rand() / RAND_MAX - 0.5f) * params->noise_level;
                pocket_state.overlays[i].values[j] += noise;
                pocket_state.overlays[i].values[j] = fmaxf(0.0f, fminf(1.0f,
                    pocket_state.overlays[i].values[j]));
            }
        }


        pocket_state.global_coherence = pocket_state.global_coherence * params->stability_bias +
                                       (1.0f - params->stability_bias) * 0.5f;

        total_coherence += pocket_state.global_coherence;
        total_decoherence += pocket_state.decoherence_level;


        if (telemetry && tick % POCKET_TELEMETRY_INTERVAL == 0) {
            fprintf(telemetry, "%d,%.4f,%.6f,%.4f,%.4f,%.4f\n",
                   tick, pocket_state.global_coherence, pocket_state.decoherence_level,
                   pocket_state.overlays[0].stability,
                   pocket_state.overlays[1].stability,
                   pocket_state.overlays[2].stability);
        }
    }

    if (telemetry) fclose(telemetry);

    clock_t end = clock();


    result->pocket_id = params->pocket_id;
    memcpy(&result->final_state, &pocket_state, sizeof(qallow_state_t));
    result->avg_coherence = total_coherence / num_ticks;
    result->avg_decoherence = total_decoherence / num_ticks;
    result->ticks_executed = num_ticks;
    result->elapsed_time_ms = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;


    ethics_state_t ethics;
    qallow_ethics_check(&pocket_state, &ethics);
    result->ethics_score = ethics.total_ethics_score;
    result->confidence = 1.0f - result->avg_decoherence;
}


void multi_pocket_execute_cpu(multi_pocket_scheduler_t* scheduler,
                              const qallow_state_t* initial_state,
                              int num_ticks) {
    if (!scheduler || !initial_state) return;

    printf("[MULTI-POCKET] Executing %d pockets on CPU...\n", scheduler->num_pockets);

    clock_t total_start = clock();
    scheduler->max_pocket_time_ms = 0.0;
    scheduler->min_pocket_time_ms = 999999.0;

    for (int i = 0; i < scheduler->num_pockets; i++) {
        printf("[POCKET %d] Starting simulation...\n", i);

        execute_single_pocket_cpu(&scheduler->params[i],
                                 initial_state,
                                 num_ticks,
                                 &scheduler->results[i]);


        if (scheduler->results[i].elapsed_time_ms > scheduler->max_pocket_time_ms) {
            scheduler->max_pocket_time_ms = scheduler->results[i].elapsed_time_ms;
        }
        if (scheduler->results[i].elapsed_time_ms < scheduler->min_pocket_time_ms) {
            scheduler->min_pocket_time_ms = scheduler->results[i].elapsed_time_ms;
        }

        printf("[POCKET %d] Complete - %.2f ms - Coherence: %.4f\n",
               i, scheduler->results[i].elapsed_time_ms,
               scheduler->results[i].avg_coherence);
    }

    clock_t total_end = clock();
    scheduler->total_scheduler_time_ms = (double)(total_end - total_start) / CLOCKS_PER_SEC * 1000.0;

    printf("[MULTI-POCKET] All pockets complete - Total: %.2f ms\n",
           scheduler->total_scheduler_time_ms);
}

#if CUDA_ENABLED

void multi_pocket_execute_cuda(multi_pocket_scheduler_t* scheduler,
                               const qallow_state_t* initial_state,
                               int num_ticks) {
    if (!scheduler || !initial_state) return;

    if (scheduler->num_pockets <= 0) {
        return;
    }

    printf("[MULTI-POCKET] Executing %d pockets on CUDA (parallel kernels)...\n",
           scheduler->num_pockets);

    int node_count = initial_state->overlays[0].node_count;
    if (node_count <= 0 || node_count > MAX_NODES) {
        node_count = MAX_NODES;
    }

    const size_t state_elements =
        (size_t)scheduler->num_pockets * NUM_OVERLAYS * MAX_NODES;
    float* final_values = (float*)malloc(state_elements * sizeof(float));
    float* avg_coherence = (float*)malloc(sizeof(float) * scheduler->num_pockets);
    float* avg_deco = (float*)malloc(sizeof(float) * scheduler->num_pockets);
    float* final_coherence = (float*)malloc(sizeof(float) * scheduler->num_pockets);

    if (!final_values || !avg_coherence || !avg_deco || !final_coherence) {
        fprintf(stderr, "[MULTI-POCKET][CUDA] Allocation failure, falling back to CPU path.\n");
        free(final_values);
        free(avg_coherence);
        free(avg_deco);
        free(final_coherence);
        multi_pocket_execute_cpu(scheduler, initial_state, num_ticks);
        return;
    }

    double total_elapsed_ms = 0.0;
    int rc = multi_pocket_cuda_run(scheduler->params,
                                   initial_state,
                                   scheduler->num_pockets,
                                   num_ticks,
                                   node_count,
                                   final_values,
                                   avg_coherence,
                                   avg_deco,
                                   final_coherence,
                                   &total_elapsed_ms);
    if (rc != 0) {
        fprintf(stderr, "[MULTI-POCKET][CUDA] Device execution failed, using CPU fallback.\n");
        free(final_values);
        free(avg_coherence);
        free(avg_deco);
        free(final_coherence);
        multi_pocket_execute_cpu(scheduler, initial_state, num_ticks);
        return;
    }

    scheduler->total_scheduler_time_ms = total_elapsed_ms;
    scheduler->max_pocket_time_ms = 0.0;
    scheduler->min_pocket_time_ms = total_elapsed_ms;

    for (int pocket = 0; pocket < scheduler->num_pockets; ++pocket) {
        qallow_state_t pocket_state;
        memcpy(&pocket_state, initial_state, sizeof(qallow_state_t));
        pocket_state.tick_count = num_ticks;
        pocket_state.global_coherence = final_coherence[pocket];
        pocket_state.decoherence_level = avg_deco[pocket];

        for (int overlay = 0; overlay < NUM_OVERLAYS; ++overlay) {
            const size_t base =
                ((size_t)pocket * NUM_OVERLAYS + (size_t)overlay) * MAX_NODES;
            memcpy(pocket_state.overlays[overlay].values,
                   final_values + base,
                   node_count * sizeof(float));
            pocket_state.overlays[overlay].node_count = node_count;
            pocket_state.overlays[overlay].stability =
                qallow_calculate_stability(&pocket_state.overlays[overlay]);
        }

        ethics_state_t ethics;
        qallow_ethics_check(&pocket_state, &ethics);

        pocket_result_t* result = &scheduler->results[pocket];
        result->pocket_id = scheduler->params[pocket].pocket_id;
        memcpy(&result->final_state, &pocket_state, sizeof(qallow_state_t));
        result->avg_coherence = avg_coherence[pocket];
        result->avg_decoherence = avg_deco[pocket];
        result->ethics_score = ethics.total_ethics_score;
        result->confidence = fmaxf(0.0f, 1.0f - result->avg_decoherence);
        result->ticks_executed = num_ticks;
        result->elapsed_time_ms = total_elapsed_ms;

        if (result->elapsed_time_ms > scheduler->max_pocket_time_ms) {
            scheduler->max_pocket_time_ms = result->elapsed_time_ms;
        }
        if (result->elapsed_time_ms < scheduler->min_pocket_time_ms) {
            scheduler->min_pocket_time_ms = result->elapsed_time_ms;
        }

        printf("[POCKET %d][CUDA] Avg coherence=%.4f | Avg deco=%.6f | Ethics=%.3f\n",
               pocket,
               result->avg_coherence,
               result->avg_decoherence,
               result->ethics_score);
    }

    printf("[MULTI-POCKET][CUDA] All pockets complete - GPU time: %.2f ms\n",
           scheduler->total_scheduler_time_ms);

    free(final_values);
    free(avg_coherence);
    free(avg_deco);
    free(final_coherence);
}
#endif


void multi_pocket_execute_all(multi_pocket_scheduler_t* scheduler,
                              const qallow_state_t* initial_state,
                              int num_ticks) {
    if (!scheduler || !initial_state) return;

#if CUDA_ENABLED
    multi_pocket_execute_cuda(scheduler, initial_state, num_ticks);
#else
    multi_pocket_execute_cpu(scheduler, initial_state, num_ticks);
#endif
}


void multi_pocket_merge(multi_pocket_scheduler_t* scheduler,
                       qallow_state_t* merged_state,
                       const pocket_merge_config_t* config) {
    if (!scheduler || !merged_state || scheduler->num_pockets == 0) return;


    bool is_outlier[MAX_POCKETS] = {false};
    if (config && config->filter_outliers) {
        multi_pocket_find_outliers(scheduler, is_outlier, config->outlier_threshold);
    }


    memset(merged_state, 0, sizeof(qallow_state_t));

    float total_weight = 0.0f;
    int valid_pockets = 0;


    for (int i = 0; i < scheduler->num_pockets; i++) {
        if (is_outlier[i]) continue;

        float weight = 1.0f;
        if (config && config->use_weighted_merge) {
            weight = scheduler->results[i].confidence;
            if (config->confidence_weight > 0.0f) {
                weight = powf(weight, config->confidence_weight);
            }
        }

        total_weight += weight;
        valid_pockets++;


        for (int j = 0; j < NUM_OVERLAYS; j++) {
            for (int k = 0; k < MAX_NODES; k++) {
                merged_state->overlays[j].values[k] +=
                    scheduler->results[i].final_state.overlays[j].values[k] * weight;
            }
        }


        merged_state->global_coherence += scheduler->results[i].avg_coherence * weight;
        merged_state->decoherence_level += scheduler->results[i].avg_decoherence * weight;
    }


    if (total_weight > 0.0f) {
        for (int j = 0; j < NUM_OVERLAYS; j++) {
            for (int k = 0; k < MAX_NODES; k++) {
                merged_state->overlays[j].values[k] /= total_weight;
            }
        }
        merged_state->global_coherence /= total_weight;
        merged_state->decoherence_level /= total_weight;
    }

    printf("[MERGE] Combined %d/%d pockets (weight=%.2f)\n",
           valid_pockets, scheduler->num_pockets, total_weight);
}


float multi_pocket_calculate_consensus(const multi_pocket_scheduler_t* scheduler) {
    if (!scheduler || scheduler->num_pockets == 0) return 0.0f;


    float mean = 0.0f;
    for (int i = 0; i < scheduler->num_pockets; i++) {
        mean += scheduler->results[i].avg_coherence;
    }
    mean /= scheduler->num_pockets;

    float variance = 0.0f;
    for (int i = 0; i < scheduler->num_pockets; i++) {
        float diff = scheduler->results[i].avg_coherence - mean;
        variance += diff * diff;
    }
    variance /= scheduler->num_pockets;


    float std_dev = sqrtf(variance);
    float consensus = 1.0f - fminf(std_dev * 5.0f, 1.0f);  // Scale std_dev

    return consensus;
}


void multi_pocket_find_outliers(const multi_pocket_scheduler_t* scheduler,
                                bool* is_outlier,
                                float threshold) {
    if (!scheduler || !is_outlier) return;


    float mean = 0.0f;
    for (int i = 0; i < scheduler->num_pockets; i++) {
        mean += scheduler->results[i].avg_coherence;
    }
    mean /= scheduler->num_pockets;


    for (int i = 0; i < scheduler->num_pockets; i++) {
        float deviation = fabsf(scheduler->results[i].avg_coherence - mean);
        is_outlier[i] = (deviation > threshold);

        if (is_outlier[i]) {
            printf("[OUTLIER] Pocket %d - Coherence %.4f deviates %.4f from mean %.4f\n",
                   i, scheduler->results[i].avg_coherence, deviation, mean);
        }
    }
}


void multi_pocket_print_results(const multi_pocket_scheduler_t* scheduler) {
    if (!scheduler) return;

    printf("\n=== MULTI-POCKET RESULTS ===\n");
    printf("Pocket | Coherence | Decoherence | Ethics | Confidence | Time(ms)\n");
    printf("-------|-----------|-------------|--------|------------|----------\n");

    for (int i = 0; i < scheduler->num_pockets; i++) {
        const pocket_result_t* r = &scheduler->results[i];
        printf("  %2d   |   %.4f   |   %.6f    | %.4f |   %.4f     | %7.2f\n",
               r->pocket_id, r->avg_coherence, r->avg_decoherence,
               r->ethics_score, r->confidence, r->elapsed_time_ms);
    }

    printf("\n");
}


void multi_pocket_print_statistics(const multi_pocket_scheduler_t* scheduler) {
    if (!scheduler) return;

    float consensus = multi_pocket_calculate_consensus(scheduler);

    printf("\n=== MULTI-POCKET STATISTICS ===\n");
    printf("Total pockets:       %d\n", scheduler->num_pockets);
    printf("Total time:          %.2f ms\n", scheduler->total_scheduler_time_ms);
    printf("Min pocket time:     %.2f ms\n", scheduler->min_pocket_time_ms);
    printf("Max pocket time:     %.2f ms\n", scheduler->max_pocket_time_ms);
    printf("Avg pocket time:     %.2f ms\n",
           scheduler->total_scheduler_time_ms / scheduler->num_pockets);
    printf("Consensus metric:    %.4f\n", consensus);
    printf("\n");
}


void multi_pocket_write_summary(const multi_pocket_scheduler_t* scheduler) {
    if (!scheduler) return;

    FILE* f = fopen("multi_pocket_summary.txt", "w");
    if (!f) return;

    fprintf(f, "Multi-Pocket Simulation Summary\n");
    fprintf(f, "================================\n\n");
    fprintf(f, "Number of pockets: %d\n", scheduler->num_pockets);
    fprintf(f, "Total runtime: %.2f ms\n\n", scheduler->total_scheduler_time_ms);

    fprintf(f, "Pocket Results:\n");
    for (int i = 0; i < scheduler->num_pockets; i++) {
        const pocket_result_t* r = &scheduler->results[i];
        fprintf(f, "  Pocket %d:\n", i);
        fprintf(f, "    Avg Coherence:   %.4f\n", r->avg_coherence);
        fprintf(f, "    Avg Decoherence: %.6f\n", r->avg_decoherence);
        fprintf(f, "    Ethics Score:    %.4f\n", r->ethics_score);
        fprintf(f, "    Confidence:      %.4f\n", r->confidence);
        fprintf(f, "    Runtime:         %.2f ms\n\n", r->elapsed_time_ms);
    }

    fprintf(f, "Consensus: %.4f\n", multi_pocket_calculate_consensus(scheduler));

    fclose(f);
    printf("[MULTI-POCKET] Summary written to multi_pocket_summary.txt\n");
}
