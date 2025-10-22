#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

// CUDA kernel for parallel sigmoid prediction
__global__ void cuda_predict_kernel(double *energy, double *risk, double *reward, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double x = energy[idx] - risk[idx];
        reward[idx] = 1.0 / (1.0 + exp(-6.0 * x)) - 0.5;
    }
}

// CUDA kernel for parallel learning updates
__global__ void cuda_learn_kernel(double *energy, double *risk, double *reward, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double target = 0.25;
        double err = target - reward[idx];
        energy[idx] += 0.02 * err;
        risk[idx] -= 0.02 * err;
    }
}

// CUDA kernel for parallel emotion regulation
__global__ void cuda_emotion_kernel(double *energy, double *risk, double *reward, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (risk[idx] > 0.8) reward[idx] -= 0.1;
        energy[idx] = fmax(0.0, fmin(1.0, energy[idx]));
        risk[idx] = fmax(-0.1, fmin(1.0, risk[idx]));
    }
}

// Wrapper: Parallel prediction on GPU
int cuda_predict_batch(double *h_energy, double *h_risk, double *h_reward, int batch_size) {
    double *d_energy, *d_risk, *d_reward;
    size_t bytes = batch_size * sizeof(double);
    
    cudaMalloc(&d_energy, bytes);
    cudaMalloc(&d_risk, bytes);
    cudaMalloc(&d_reward, bytes);
    
    cudaMemcpy(d_energy, h_energy, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_risk, h_risk, bytes, cudaMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    cuda_predict_kernel<<<blocks, threads>>>(d_energy, d_risk, d_reward, batch_size);
    
    cudaMemcpy(h_reward, d_reward, bytes, cudaMemcpyDeviceToHost);
    
    cudaFree(d_energy);
    cudaFree(d_risk);
    cudaFree(d_reward);
    
    return 0;
}

// Wrapper: Parallel learning on GPU
int cuda_learn_batch(double *h_energy, double *h_risk, double *h_reward, int batch_size) {
    double *d_energy, *d_risk, *d_reward;
    size_t bytes = batch_size * sizeof(double);
    
    cudaMalloc(&d_energy, bytes);
    cudaMalloc(&d_risk, bytes);
    cudaMalloc(&d_reward, bytes);
    
    cudaMemcpy(d_energy, h_energy, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_risk, h_risk, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_reward, h_reward, bytes, cudaMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    cuda_learn_kernel<<<blocks, threads>>>(d_energy, d_risk, d_reward, batch_size);
    
    cudaMemcpy(h_energy, d_energy, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_risk, d_risk, bytes, cudaMemcpyDeviceToHost);
    
    cudaFree(d_energy);
    cudaFree(d_risk);
    cudaFree(d_reward);
    
    return 0;
}

// Wrapper: Parallel emotion regulation on GPU
int cuda_emotion_batch(double *h_energy, double *h_risk, double *h_reward, int batch_size) {
    double *d_energy, *d_risk, *d_reward;
    size_t bytes = batch_size * sizeof(double);
    
    cudaMalloc(&d_energy, bytes);
    cudaMalloc(&d_risk, bytes);
    cudaMalloc(&d_reward, bytes);
    
    cudaMemcpy(d_energy, h_energy, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_risk, h_risk, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_reward, h_reward, bytes, cudaMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    cuda_emotion_kernel<<<blocks, threads>>>(d_energy, d_risk, d_reward, batch_size);
    
    cudaMemcpy(h_energy, d_energy, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_risk, d_risk, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_reward, d_reward, bytes, cudaMemcpyDeviceToHost);
    
    cudaFree(d_energy);
    cudaFree(d_risk);
    cudaFree(d_reward);
    
    return 0;
}

