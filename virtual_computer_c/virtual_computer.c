#include "virtual_computer.h"
#include <stdio.h>

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

    return 0;
}

void vc_free(VirtualComputer *vc) {
    if (!vc) {
        return;
    }
    photonic_free(&vc->photon);
    neuro_free(&vc->neuro);
    cuda_gpu_free(&vc->gpu);
}

int vc_run_task(VirtualComputer *vc, const Workload *task) {
    if (!vc || !task) {
        return -1;
    }

    switch (task->type) {
        case TASK_NUMERIC: {
            CudaLaunchCfg cfg = {.grid_x = 64, .block_x = 128};
            cuda_gpu_launch_kernel(&vc->gpu, cfg, task->payload, task->payload_size);
            printf("[GPU] kernels=%llu last=%.3f ms\n",
                   (unsigned long long)vc->gpu.perf.kernels_launched,
                   cuda_gpu_last_exec_ms(&vc->gpu));
            break;
        }
        case TASK_SPIKING: {
            neuro_inject_spike(&vc->neuro, 0);
            neuro_step(&vc->neuro, 10);
            printf("[NEURO] steps=%llu spikes=%llu\n",
                   (unsigned long long)vc->neuro.perf.steps,
                   (unsigned long long)vc->neuro.perf.spikes);
            break;
        }
        case TASK_PHOTONIC: {
            photonic_inject(&vc->photon, 0, 1.0f, 0.0f);
            photonic_inject(&vc->photon, 1, 0.5f, 0.0f);
            photonic_propagate(&vc->photon, 5);
            printf("[PHOTON] propagations=%llu\n",
                   (unsigned long long)vc->photon.perf.propagations);
            break;
        }
        default:
            return -2;
    }

    return 0;
}
