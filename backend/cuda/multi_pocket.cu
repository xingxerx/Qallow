#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <vector>

#include "multi_pocket.h"

#define CUDA_CHECK(expr)                                      \
    do {                                                      \
        cudaError_t err__ = (expr);                           \
        if (err__ != cudaSuccess) {                           \
            fprintf(stderr, "[MULTI-POCKET][CUDA] %s failed: %s\n", \
                    #expr, cudaGetErrorString(err__));        \
            return -1;                                        \
        }                                                     \
    } while (0)

namespace {

constexpr int kOverlays = NUM_OVERLAYS;
constexpr int kMaxNodes = MAX_NODES;

struct DevicePocketParams {
    float learning_rate;
    float noise_level;
    float stability_bias;
    float padding;  // keep 16-byte alignment
};

__device__ inline float hash_uniform(unsigned long long x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    const unsigned long long mant = x & 0x00FFFFFFULL;
    return static_cast<float>(mant) * (1.0f / 16777216.0f);  // 2^24
}

__global__ void zero_tick_buffers(float* tick_sum,
                                  float* tick_deco,
                                  int num_pockets) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pockets) {
        tick_sum[idx] = 0.0f;
        tick_deco[idx] = 0.0f;
    }
}

__global__ void pocket_tick_kernel(float* values,
                                   const DevicePocketParams* params,
                                   float* tick_sum,
                                   float* tick_deco,
                                   int num_pockets,
                                   int node_count,
                                   int tick,
                                   unsigned long long seed) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    int combo = blockIdx.y;
    if (node >= node_count || combo >= num_pockets * kOverlays) {
        return;
    }

    int pocket = combo / kOverlays;
    int overlay = combo % kOverlays;
    const DevicePocketParams cfg = params[pocket];

    const int idx = ((pocket * kOverlays) + overlay) * kMaxNodes + node;
    float value = values[idx];


    const float base_targets[kOverlays] = {0.72f, 0.68f, 0.75f};
    float target = base_targets[overlay];


    value += cfg.learning_rate * (target - value);


    unsigned long long noise_seed = seed;
    noise_seed ^= static_cast<unsigned long long>(pocket) << 40;
    noise_seed ^= static_cast<unsigned long long>(overlay) << 32;
    noise_seed ^= static_cast<unsigned long long>(node) << 16;
    noise_seed ^= static_cast<unsigned long long>(tick);
    float rand_unit = hash_uniform(noise_seed);
    float centered = rand_unit - 0.5f;
    float noise = centered * cfg.noise_level;

    value = fmaf(noise, 1.0f, value);
    value = fminf(1.0f, fmaxf(0.0f, value));
    values[idx] = value;

    atomicAdd(&tick_sum[pocket], value);
    atomicAdd(&tick_deco[pocket], fabsf(noise) * 0.1f + 0.0001f);
}

__global__ void finalize_tick_kernel(float* tick_sum,
                                     float* tick_deco,
                                     float* sum_coherence,
                                     float* sum_decoherence,
                                     float* current_coherence,
                                     const DevicePocketParams* params,
                                     int num_pockets,
                                     int node_count,
                                     int num_ticks_completed) {
    int pocket = blockIdx.x * blockDim.x + threadIdx.x;
    if (pocket >= num_pockets) {
        return;
    }

    const float denom = static_cast<float>(node_count * kOverlays);
    float avg_activation = tick_sum[pocket] / denom;
    float avg_deco = tick_deco[pocket] / denom;

    float updated = current_coherence[pocket] * params[pocket].stability_bias +
                    (1.0f - params[pocket].stability_bias) * avg_activation;

    current_coherence[pocket] = updated;
    sum_coherence[pocket] += updated;
    sum_decoherence[pocket] += avg_deco;


    tick_sum[pocket] = 0.0f;
    tick_deco[pocket] = 0.0f;
}

__global__ void compute_results_kernel(const float* sum_coherence,
                                       const float* sum_decoherence,
                                       const float* current_coherence,
                                       float* avg_coherence_out,
                                       float* avg_deco_out,
                                       float* final_coherence_out,
                                       int num_pockets,
                                       int num_ticks) {
    int pocket = blockIdx.x * blockDim.x + threadIdx.x;
    if (pocket >= num_pockets) {
        return;
    }
    const float inv_ticks = (num_ticks > 0) ? (1.0f / static_cast<float>(num_ticks)) : 0.0f;
    avg_coherence_out[pocket] = sum_coherence[pocket] * inv_ticks;
    avg_deco_out[pocket] = sum_decoherence[pocket] * inv_ticks;
    final_coherence_out[pocket] = current_coherence[pocket];
}

}  // namespace

extern "C" int multi_pocket_cuda_run(const pocket_params_t* params,
                                     const qallow_state_t* initial_state,
                                     int num_pockets,
                                     int num_ticks,
                                     int node_count,
                                     float* host_values_out,
                                     float* avg_coherence_out,
                                     float* avg_deco_out,
                                     float* final_coherence_out,
                                     double* total_elapsed_ms) {
    if (!params || !initial_state || !host_values_out || !avg_coherence_out ||
        !avg_deco_out || !final_coherence_out || !total_elapsed_ms) {
        return -1;
    }
    if (num_pockets <= 0 || num_pockets > MAX_POCKETS) {
        return -1;
    }
    if (node_count <= 0 || node_count > kMaxNodes) {
        return -1;
    }
    if (num_ticks <= 0) {
        return -1;
    }

    const size_t state_elements = static_cast<size_t>(num_pockets) * kOverlays * kMaxNodes;
    const size_t state_bytes = state_elements * sizeof(float);

    std::vector<float> init_values(state_elements, 0.0f);
    for (int pocket = 0; pocket < num_pockets; ++pocket) {
        for (int overlay = 0; overlay < kOverlays; ++overlay) {
            const overlay_t& src = initial_state->overlays[overlay];
            const size_t base = ((size_t)pocket * kOverlays + (size_t)overlay) * kMaxNodes;
            memcpy(init_values.data() + base, src.values, node_count * sizeof(float));
        }
    }

    std::vector<DevicePocketParams> host_params(num_pockets);
    for (int pocket = 0; pocket < num_pockets; ++pocket) {
        host_params[pocket].learning_rate = params[pocket].learning_rate;
        host_params[pocket].noise_level = params[pocket].noise_level;
        host_params[pocket].stability_bias = params[pocket].stability_bias;
        host_params[pocket].padding = 0.0f;
    }

    std::vector<float> host_current(num_pockets, initial_state->global_coherence);

    float* d_values = nullptr;
    DevicePocketParams* d_params = nullptr;
    float *d_tick_sum = nullptr, *d_tick_deco = nullptr;
    float *d_sum_coherence = nullptr, *d_sum_deco = nullptr;
    float* d_current_coherence = nullptr;
    float *d_avg_coherence = nullptr, *d_avg_deco = nullptr, *d_final_coherence = nullptr;

    CUDA_CHECK(cudaMalloc(&d_values, state_bytes));
    CUDA_CHECK(cudaMemcpy(d_values, init_values.data(), state_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_params, sizeof(DevicePocketParams) * num_pockets));
    CUDA_CHECK(cudaMemcpy(d_params, host_params.data(),
                          sizeof(DevicePocketParams) * num_pockets,
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_tick_sum, sizeof(float) * num_pockets));
    CUDA_CHECK(cudaMalloc(&d_tick_deco, sizeof(float) * num_pockets));
    CUDA_CHECK(cudaMalloc(&d_sum_coherence, sizeof(float) * num_pockets));
    CUDA_CHECK(cudaMalloc(&d_sum_deco, sizeof(float) * num_pockets));
    CUDA_CHECK(cudaMalloc(&d_current_coherence, sizeof(float) * num_pockets));
    CUDA_CHECK(cudaMalloc(&d_avg_coherence, sizeof(float) * num_pockets));
    CUDA_CHECK(cudaMalloc(&d_avg_deco, sizeof(float) * num_pockets));
    CUDA_CHECK(cudaMalloc(&d_final_coherence, sizeof(float) * num_pockets));

    CUDA_CHECK(cudaMemcpy(d_current_coherence, host_current.data(),
                          sizeof(float) * num_pockets, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_sum_coherence, 0, sizeof(float) * num_pockets));
    CUDA_CHECK(cudaMemset(d_sum_deco, 0, sizeof(float) * num_pockets));
    CUDA_CHECK(cudaMemset(d_tick_sum, 0, sizeof(float) * num_pockets));
    CUDA_CHECK(cudaMemset(d_tick_deco, 0, sizeof(float) * num_pockets));

    cudaEvent_t start_evt, end_evt;
    CUDA_CHECK(cudaEventCreate(&start_evt));
    CUDA_CHECK(cudaEventCreate(&end_evt));
    CUDA_CHECK(cudaEventRecord(start_evt, 0));

    dim3 zero_grid((num_pockets + 127) / 128);
    dim3 zero_block(128);
    dim3 update_block(128);
    dim3 update_grid((node_count + update_block.x - 1) / update_block.x,
                     num_pockets * kOverlays);
    dim3 finalize_block(128);
    dim3 finalize_grid((num_pockets + finalize_block.x - 1) / finalize_block.x);

    const unsigned long long base_seed = 0x9E3779B97F4A7C15ULL;
    for (int tick = 0; tick < num_ticks; ++tick) {
        zero_tick_buffers<<<zero_grid, zero_block>>>(d_tick_sum, d_tick_deco, num_pockets);
        CUDA_CHECK(cudaGetLastError());

        pocket_tick_kernel<<<update_grid, update_block>>>(
            d_values, d_params, d_tick_sum, d_tick_deco,
            num_pockets, node_count, tick, base_seed + tick * 1315423911ULL);
        CUDA_CHECK(cudaGetLastError());

        finalize_tick_kernel<<<finalize_grid, finalize_block>>>(
            d_tick_sum, d_tick_deco, d_sum_coherence, d_sum_deco,
            d_current_coherence, d_params, num_pockets, node_count, tick + 1);
        CUDA_CHECK(cudaGetLastError());
    }

    compute_results_kernel<<<finalize_grid, finalize_block>>>(
        d_sum_coherence, d_sum_deco, d_current_coherence,
        d_avg_coherence, d_avg_deco, d_final_coherence,
        num_pockets, num_ticks);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(end_evt, 0));
    CUDA_CHECK(cudaEventSynchronize(end_evt));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_evt, end_evt));
    *total_elapsed_ms = static_cast<double>(elapsed_ms);

    CUDA_CHECK(cudaMemcpy(host_values_out, d_values, state_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(avg_coherence_out, d_avg_coherence,
                          sizeof(float) * num_pockets, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(avg_deco_out, d_avg_deco,
                          sizeof(float) * num_pockets, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(final_coherence_out, d_final_coherence,
                          sizeof(float) * num_pockets, cudaMemcpyDeviceToHost));

    cudaEventDestroy(start_evt);
    cudaEventDestroy(end_evt);
    cudaFree(d_values);
    cudaFree(d_params);
    cudaFree(d_tick_sum);
    cudaFree(d_tick_deco);
    cudaFree(d_sum_coherence);
    cudaFree(d_sum_deco);
    cudaFree(d_current_coherence);
    cudaFree(d_avg_coherence);
    cudaFree(d_avg_deco);
    cudaFree(d_final_coherence);
    return 0;
}
