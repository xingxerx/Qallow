#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
extern "C" {
#include "qallow.h"
}

__global__ void quantumOptimize(double* data, int n, double step, double target_center){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        // simple energy-reduction shaping around 0.5
        double x = data[i];
        double grad = (target_center - x);
        data[i] = x + step * grad; // adaptive step toward logical midline
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

__global__ void surfaceCodeAverage(double* data,
                                   int n,
                                   int block_size,
                                   double correction_weight,
                                   double logical_error_rate) {
    int logical_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int start = logical_idx * block_size;
    if (start >= n) {
        return;
    }

    double sum = 0.0;
    int count = 0;
    for (int offset = 0; offset < block_size && (start + offset) < n; ++offset) {
        sum += data[start + offset];
        ++count;
    }
    if (count == 0) {
        return;
    }

    double mean = sum / (double)count;
    double logical_bias = 0.5 + (mean - 0.5) * (1.0 - logical_error_rate);
    double corrected = mean + correction_weight * (logical_bias - mean);
    if (corrected < 0.0) corrected = 0.0;
    if (corrected > 1.0) corrected = 1.0;

    for (int offset = 0; offset < count; ++offset) {
        double physical = data[start + offset];
        double blended = physical * (1.0 - correction_weight) + corrected * correction_weight;
        if (blended < 0.0) blended = 0.0;
        if (blended > 1.0) blended = 1.0;
        data[start + offset] = blended;
    }
}

extern "C" void runQuantumOptimizer(double* hostData, int n){
    if (n <= 0) {
        return;
    }

    double* d = nullptr;
    if (cudaMalloc(&d, n * sizeof(double)) != cudaSuccess) {
        return;
    }
    cudaMemcpy(d, hostData, n * sizeof(double), cudaMemcpyHostToDevice);

    const int t = 256;
    const int b = (n + t - 1) / t;

    const char* distance_env = getenv("QALLOW_SURFACE_CODE_DISTANCE");
    const char* block_env = getenv("QALLOW_LOGICAL_BLOCK_SIZE");
    const char* error_env = getenv("QALLOW_PHYSICAL_ERROR_RATE");

    int surface_distance = 1;
    if (distance_env && *distance_env) {
        int parsed = atoi(distance_env);
        if (parsed > 0) {
            surface_distance = parsed;
        }
    }
    if (surface_distance < 1) {
        surface_distance = 1;
    }

    int block_size = 1;
    if (surface_distance > 1) {
        long long block_calc = (long long)surface_distance * (long long)surface_distance;
        if (block_calc > INT_MAX) {
            block_size = INT_MAX;
        } else {
            block_size = (int)block_calc;
        }
    }
    if (block_env && *block_env) {
        int override_block = atoi(block_env);
        if (override_block > 0) {
            block_size = override_block;
        }
    }
    if (block_size < 1) {
        block_size = 1;
    }

    double physical_error = 0.01;
    if (error_env && *error_env) {
        double parsed = atof(error_env);
        if (parsed >= 0.0) {
            physical_error = parsed;
        }
    }
    if (physical_error > 1.0) {
        physical_error = 1.0;
    }

    double distance_factor = surface_distance > 0 ? ((double)surface_distance / 2.0) : 0.5;
    double logical_error_rate = pow(physical_error, distance_factor);
    if (!isfinite(logical_error_rate) || logical_error_rate < 0.0) {
        logical_error_rate = 0.0;
    }
    if (logical_error_rate > 1.0) {
        logical_error_rate = 1.0;
    }

    double stabilizer_gain = 1.0 - logical_error_rate * 12.0;
    if (stabilizer_gain < 0.1) {
        stabilizer_gain = 0.1;
    }
    if (stabilizer_gain > 0.8) {
        stabilizer_gain = 0.8;
    }

    double gradient_step = 0.08 * stabilizer_gain;
    double correction_weight = 0.25 * stabilizer_gain;

    quantumOptimize<<<b, t>>>(d, n, gradient_step, 0.5);
    cudaDeviceSynchronize();

    if (block_size > 1) {
        int logical_blocks = (n + block_size - 1) / block_size;
        int block_threads = 256;
        int block_grid = (logical_blocks + block_threads - 1) / block_threads;
        surfaceCodeAverage<<<block_grid, block_threads>>>(d, n, block_size, correction_weight, logical_error_rate);
        cudaDeviceSynchronize();
    }

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

