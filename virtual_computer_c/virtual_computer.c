#include "virtual_computer.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static void vc_update_temporal_memory(VirtualComputer *vc,
                                      TaskType type,
                                      float signal_metric,
                                      const char *label);

int vc_init(VirtualComputer *vc) {
    if (!vc) {
        return -1;
    }

    if (cuda_gpu_init(&vc->gpu, 0) != 0) {
        return -1;
    }
    if (cuda_gpu_alloc(&vc->gpu, 1 << 20) != 0) {
        return -1;
    }

    if (neuro_init(&vc->neuro, 256, 512, 0.5f) != 0) {
        return -1;
    }
    for (int i = 0; i < 10; ++i) {
        (void)neuro_connect(&vc->neuro, i, i + 1, 1.0f, 1.0f);
    }

    if (photonic_init(&vc->photon, 16, 32) != 0) {
        return -1;
    }
    PhotonicGate gate = {
        .type = PHOTONIC_GATE_BS,
        .in_a = 0,
        .in_b = 1,
        .out_a = 2,
        .out_b = 3,
        .theta = 0.0f,
        .eta = 0.5f
    };
    (void)photonic_add_gate(&vc->photon, gate);

    if (tm_init(&vc->memory, 0.2f) != 0) {
        return -1;
    }

    return 0;
}

void vc_free(VirtualComputer *vc) {
    if (!vc) {
        return;
    }
    photonic_free(&vc->photon);
    neuro_free(&vc->neuro);
    cuda_gpu_free(&vc->gpu);
    tm_free(&vc->memory);
}

int vc_run_task(VirtualComputer *vc, const Workload *task) {
    if (!vc || !task) {
        return -1;
    }

    switch (task->type) {
        case TASK_NUMERIC: {
            CudaLaunchCfg cfg = {.grid_x = 64, .block_x = 128};
            cuda_gpu_launch_kernel(&vc->gpu, cfg, task->payload, task->payload_size);
            double last_ms = cuda_gpu_last_exec_ms(&vc->gpu);
            printf("[GPU] kernels=%llu last=%.3f ms\n",
                   (unsigned long long)vc->gpu.perf.kernels_launched,
                   last_ms);
            float normalized = (float)fmin(fmax(last_ms / 10.0, 0.0), 1.0);
            vc_update_temporal_memory(vc, TASK_NUMERIC, normalized, "numeric_kernel");
            break;
        }
        case TASK_SPIKING: {
            neuro_inject_spike(&vc->neuro, 0);
            neuro_step(&vc->neuro, 10);
            float steps = (float)vc->neuro.perf.steps;
            float normalized = fminf(steps / 1000.0f, 1.0f);
            printf("[NEURO] steps=%llu spikes=%llu\n",
                   (unsigned long long)vc->neuro.perf.steps,
                   (unsigned long long)vc->neuro.perf.spikes);
            vc_update_temporal_memory(vc, TASK_SPIKING, normalized, "spiking_pattern");
            break;
        }
        case TASK_PHOTONIC: {
            photonic_inject(&vc->photon, 0, 1.0f, 0.0f);
            photonic_inject(&vc->photon, 1, 0.5f, 0.0f);
            photonic_propagate(&vc->photon, 5);
            float propagations = (float)vc->photon.perf.propagations;
            float normalized = fminf(propagations / 500.0f, 1.0f);
            printf("[PHOTON] propagations=%llu\n",
                   (unsigned long long)vc->photon.perf.propagations);
            vc_update_temporal_memory(vc, TASK_PHOTONIC, normalized, "photonic_sequence");
            break;
        }
        default:
            return -2;
    }

    return 0;
}

static void vc_update_temporal_memory(VirtualComputer *vc,
                                      TaskType type,
                                      float signal_metric,
                                      const char *label) {
    if (!vc) {
        return;
    }

    float embedding[TM_VECTOR_DIM] = {0.0f};
    embedding[0] = (float)type;
    embedding[1] = signal_metric;
    embedding[2] = vc->memory.avg_coherence;
    embedding[3] = tm_predict_next(&vc->memory);

    tm_store_episodic(&vc->memory, embedding, TM_VECTOR_DIM);

    float gradient_signal = signal_metric - vc->memory.avg_coherence;
    tm_update_gradient(&vc->memory, gradient_signal);

    MemoryVector *nearest = NULL;
    (void)tm_retrieve_similar(&vc->memory, embedding, TM_VECTOR_DIM, &nearest);

    float drift = 0.0f;
    int drift_status = tm_audit_drift(&vc->memory, &drift);
    if (drift_status > 0) {
        tm_recalibrate(&vc->memory);
        if (label) {
            tm_store_semantic(&vc->memory, embedding, TM_VECTOR_DIM, label);
        }
    }

    float forecast = tm_predict_next(&vc->memory);
    printf("[TM] task=%d drift=%.3f forecast=%.3f coherence=%.3f\n",
           type,
           drift,
           forecast,
           vc->memory.avg_coherence);
}
