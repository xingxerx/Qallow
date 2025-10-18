#include <cuda_runtime.h>
#include <math.h>
extern "C" {
#include "qallow.h"
}

__global__ void quantumOptimize(double* data, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        // simple energy-reduction shaping around 0.5
        double x = data[i];
        double grad = (0.5 - x);
        data[i] = x + 0.08 * grad; // small step toward midline
        if (data[i] < 0.0) data[i]=0.0;
        if (data[i] > 1.0) data[i]=1.0;
    }
}

extern "C" void runQuantumOptimizer(double* hostData, int n){
    double* d = nullptr;
    cudaMalloc(&d, n * sizeof(double));
    cudaMemcpy(d, hostData, n * sizeof(double), cudaMemcpyHostToDevice);

    int t = 256, b = (n + t - 1) / t;
    quantumOptimize<<<b, t>>>(d, n);
    cudaDeviceSynchronize();

    cudaMemcpy(hostData, d, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d);
}

