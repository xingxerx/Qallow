#include <cuda_runtime.h>
#include <math_constants.h>

#include "reduce.cuh"

__global__ void k_harm(int N, float freq, float* acc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float e = 0.0f;
    if (i < N) {
        float t = (float)i / (float)N;
        float y = __sinf(2.0f * CUDART_PI_F * freq * t);
        e = y * y;
    }
    float s = block_sum<256>(e);
    if (threadIdx.x == 0) {
        atomicAdd(acc, s);
    }
}

extern "C" int qallow_p13_harmonic_gpu(int N, float freq, float* out_energy) {
    float* d_acc = nullptr;
    cudaError_t err = cudaMalloc(&d_acc, sizeof(float));
    if (err != cudaSuccess) {
        return -1;
    }
    cudaMemset(d_acc, 0, sizeof(float));

    dim3 bs(256);
    dim3 gs((N + bs.x - 1) / bs.x);
    k_harm<<<gs, bs>>>(N, freq, d_acc);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_acc);
        return -2;
    }

    float h = 0.0f;
    cudaMemcpy(&h, d_acc, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_acc);
    *out_energy = h;
    return 0;
}
