#include <cuda_runtime.h>
#include <math.h>

#include "core/phase16_meta_introspect.cuh"

extern "C" __global__
void introspect_kernel(const float* __restrict__ dur,
                       const float* __restrict__ coh,
                       const float* __restrict__ eth,
                       float* __restrict__ out,
                       int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float duration = fmaxf(dur[i], 0.0f);
        float coherence = coh[i];
        float ethics = eth[i];
        float score = 0.4f * coherence + 0.4f * ethics + 0.2f * log1pf(duration);
        out[i] = fminf(score, 1.0f);
    }
}

extern "C" int qallow_meta_introspect_gpu(const float* durations,
                                          const float* coherence,
                                          const float* ethics,
                                          float* improvement_scores,
                                          int count) {
    if (!durations || !coherence || !ethics || !improvement_scores || count <= 0) {
        return -1;
    }

    float *d_durations = nullptr, *d_coherence = nullptr, *d_ethics = nullptr, *d_out = nullptr;
    size_t bytes = sizeof(float) * (size_t)count;

    if (cudaMalloc(&d_durations, bytes) != cudaSuccess ||
        cudaMalloc(&d_coherence, bytes) != cudaSuccess ||
        cudaMalloc(&d_ethics, bytes) != cudaSuccess ||
        cudaMalloc(&d_out, bytes) != cudaSuccess) {
        cudaFree(d_durations);
        cudaFree(d_coherence);
        cudaFree(d_ethics);
        cudaFree(d_out);
        return -2;
    }

    cudaMemcpy(d_durations, durations, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_coherence, coherence, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ethics, ethics, bytes, cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((count + block.x - 1) / block.x);
    introspect_kernel<<<grid, block>>>(d_durations, d_coherence, d_ethics, d_out, count);
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        cudaFree(d_durations);
        cudaFree(d_coherence);
        cudaFree(d_ethics);
        cudaFree(d_out);
        return -3;
    }

    cudaMemcpy(improvement_scores, d_out, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_durations);
    cudaFree(d_coherence);
    cudaFree(d_ethics);
    cudaFree(d_out);
    return 0;
}
