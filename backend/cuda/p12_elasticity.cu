#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "reduce.cuh"

__global__ void k_elastic(int N, float scale, float* acc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = 0.0f;
    if (i < N) {
        // Toy “elasticity”: iteratively pull towards mid-point
        float x = (i % 1024) * 0.001f;
        for (int t = 0; t < 32; ++t) {
            x = x + scale * (0.5f - x);
        }
        v = x;
    }
    float s = block_sum<256>(v);
    if (threadIdx.x == 0) {
        atomicAdd(acc, s);
    }
}

extern "C" int qallow_p12_elasticity_gpu(int N, float scale, float* out_mean) {
    float* d_acc = nullptr;
    cudaError_t err = cudaMalloc(&d_acc, sizeof(float));
    if (err != cudaSuccess) {
        return -1;
    }
    cudaMemset(d_acc, 0, sizeof(float));

    dim3 bs(256);
    dim3 gs((N + bs.x - 1) / bs.x);
    k_elastic<<<gs, bs>>>(N, scale, d_acc);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_acc);
        return -2;
    }

    float h = 0.0f;
    cudaMemcpy(&h, d_acc, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_acc);
    *out_mean = h / (float)N;
    return 0;
}
