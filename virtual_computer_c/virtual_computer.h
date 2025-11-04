#ifndef VIRTUAL_COMPUTER_H
#define VIRTUAL_COMPUTER_H

#include "cuda_gpu.h"
#include "neuromorphic.h"
#include "photonic.h"
#include "temporal_memory.h"
#include <stddef.h>

typedef enum {
    TASK_NUMERIC,
    TASK_SPIKING,
    TASK_PHOTONIC
} TaskType;

typedef struct {
    TaskType type;
    void *payload;
    size_t payload_size;
} Workload;

typedef struct {
    CudaGPU gpu;
    NeuromorphicProcessor neuro;
    PhotonicProcessor photon;
    TemporalMemory memory;
} VirtualComputer;

int vc_init(VirtualComputer *vc);
void vc_free(VirtualComputer *vc);
int vc_run_task(VirtualComputer *vc, const Workload *task);

#endif /* VIRTUAL_COMPUTER_H */
