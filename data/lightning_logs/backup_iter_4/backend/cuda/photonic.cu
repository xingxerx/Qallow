#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdint.h>
extern "C" {
#include "qallow.h"
}

__global__ void photonicKernel(double* out, int n, unsigned long seed){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        curandStatePhilox4_32_10_t state;
        curand_init((unsigned long long)seed, i, 0, &state);
        // emulate photon probability sample in [0,1)
        double r = curand_uniform_double(&state);
        out[i] = r;
    }
}

extern "C" void runPhotonicSimulation(double* hostData, int n, unsigned long seed){
    double* d_out = nullptr;
    cudaMalloc(&d_out, n * sizeof(double));

    int t = 256, b = (n + t - 1) / t;
    photonicKernel<<<b, t>>>(d_out, n, seed);
    cudaDeviceSynchronize();

    cudaMemcpy(hostData, d_out, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_out);
}

