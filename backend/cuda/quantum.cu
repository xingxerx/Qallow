#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
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

__global__ void lindbladDamping(double* data, int n, double gamma, double dt){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        double value = data[i];
        double decay = exp(-gamma * dt);
        double relaxed = value * decay + (1.0 - decay) * 0.5;
        if (relaxed < 0.0) relaxed = 0.0;
        if (relaxed > 1.0) relaxed = 1.0;
        data[i] = relaxed;
    }
}

extern "C" void runQuantumOptimizer(double* hostData, int n){
    double* d = nullptr;
    cudaMalloc(&d, n * sizeof(double));
    cudaMemcpy(d, hostData, n * sizeof(double), cudaMemcpyHostToDevice);

    int t = 256, b = (n + t - 1) / t;
    quantumOptimize<<<b, t>>>(d, n);
    cudaDeviceSynchronize();

    const char* gamma_env = getenv("QALLOW_LINDBLAD_GAMMA");
    const char* dt_env = getenv("QALLOW_LINDBLAD_DT");
    double gamma = 0.0;
    double dt = 0.05;
    if (gamma_env && *gamma_env) {
        gamma = atof(gamma_env);
        if (gamma < 0.0) {
            gamma = 0.0;
        }
    }
    if (dt_env && *dt_env) {
        dt = atof(dt_env);
        if (dt <= 0.0) {
            dt = 0.05;
        }
    }

    if (gamma > 0.0) {
        lindbladDamping<<<b, t>>>(d, n, gamma, dt);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(hostData, d, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d);
}

