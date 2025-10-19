#pragma once

#include <cuda_runtime.h>
#include <cstdio>

__global__ void phase_demo_kernel(float* out, int phase, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float value = (float)phase * 0.1f + (float)(idx % 13) * 0.01f;
    for (int i = 0; i < iterations; ++i) {
        value = value * 0.999f + 0.001f * (float)phase;
    }
    out[idx] = value;
}

inline int run_phase_demo_kernel(int phase_index, int ticks) {
    const int threads = 128;
    const int blocks = 4;

    float* device_output = nullptr;
    cudaError_t err = cudaMalloc(&device_output, sizeof(float) * threads * blocks);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[phase%02d] cudaMalloc failed: %s\n", phase_index, cudaGetErrorString(err));
        return -1;
    }

    phase_demo_kernel<<<blocks, threads>>>(device_output, phase_index, ticks);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[phase%02d] kernel failed: %s\n", phase_index, cudaGetErrorString(err));
        cudaFree(device_output);
        return -2;
    }

    cudaFree(device_output);
    std::printf("[phase%02d] demo complete (%d ticks)\n", phase_index, ticks);
    return 0;
}
