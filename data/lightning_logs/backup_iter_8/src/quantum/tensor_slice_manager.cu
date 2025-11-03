#include "tensor_slice_manager.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err__ = (call); \
        if (err__ != cudaSuccess) { \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
            std::abort(); \
        } \
    } while (0)

namespace {
__device__ __forceinline__ void apply_gate_pair(cuDoubleComplex gate00,
                                               cuDoubleComplex gate01,
                                               cuDoubleComplex gate10,
                                               cuDoubleComplex gate11,
                                               cuDoubleComplex* a0,
                                               cuDoubleComplex* a1) {
    cuDoubleComplex v0 = *a0;
    cuDoubleComplex v1 = *a1;
    cuDoubleComplex r0 = cuCadd(cuCmul(gate00, v0), cuCmul(gate01, v1));
    cuDoubleComplex r1 = cuCadd(cuCmul(gate10, v0), cuCmul(gate11, v1));
    *a0 = r0;
    *a1 = r1;
}

__global__ void kernel_apply_single_qubit_gate(cuDoubleComplex* state,
                                               int num_qubits,
                                               int target,
                                               cuDoubleComplex gate00,
                                               cuDoubleComplex gate01,
                                               cuDoubleComplex gate10,
                                               cuDoubleComplex gate11) {
    const size_t mask = (1ULL << target);
    const size_t total = 1ULL << num_qubits;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t base = idx; base < total / 2; base += stride) {
        size_t low = base & (mask - 1);
        size_t high = base >> target;
        size_t i0 = (high << (target + 1)) | low;
        size_t i1 = i0 | mask;
        apply_gate_pair(gate00, gate01, gate10, gate11, &state[i0], &state[i1]);
    }
}

__global__ void kernel_apply_controlled_phase(cuDoubleComplex* state,
                                              int num_qubits,
                                              int control,
                                              int target,
                                              double theta) {
    const size_t total = 1ULL << num_qubits;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    size_t control_mask = (1ULL << control);
    size_t target_mask = (1ULL << target);
    cuDoubleComplex phase = make_cuDoubleComplex(cos(theta), sin(theta));

    for (size_t i = idx; i < total; i += stride) {
        if (((i & control_mask) != 0) && ((i & target_mask) != 0)) {
            state[i] = cuCmul(state[i], phase);
        }
    }
}

__global__ void kernel_apply_cnot(cuDoubleComplex* state,
                                  int num_qubits,
                                  int control,
                                  int target) {
    const size_t total = 1ULL << num_qubits;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    size_t control_mask = (1ULL << control);
    size_t target_mask = (1ULL << target);

    for (size_t i = idx; i < total; i += stride) {
        if ((i & control_mask) != 0) {
            size_t flipped = i ^ target_mask;
            if (flipped > i) {
                cuDoubleComplex temp = state[i];
                state[i] = state[flipped];
                state[flipped] = temp;
            }
        }
    }
}

__global__ void kernel_depolarizing_noise(cuDoubleComplex* state,
                                          int num_qubits,
                                          int target,
                                          double probability,
                                          unsigned long long seed) {
    const size_t total = 1ULL << num_qubits;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    size_t mask = (1ULL << target);

    for (size_t base = idx; base < total / 2; base += stride) {
        size_t low = base & (mask - 1);
        size_t high = base >> target;
        size_t i0 = (high << (target + 1)) | low;
        size_t i1 = i0 | mask;

        unsigned long long k = seed ^ (i0 * 0x9E3779B97f4A7C15ULL) ^ (base << 1);
        double r = (double)((k >> 11) & 0x1FFFFF) / (double)(0x1FFFFF);
        if (r < probability) {
            cuDoubleComplex tmp = state[i0];
            state[i0] = state[i1];
            state[i1] = tmp;
        }
    }
}

__global__ void kernel_checkpoint(cuDoubleComplex* state,
                                  cuDoubleComplex* buffer,
                                  size_t elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < elements; i += stride) {
        buffer[i] = state[i];
    }
}

}  // namespace

extern "C" {

QuantumSlice* qs_create(int num_qubits) {
    if (num_qubits <= 0 || num_qubits > 32) {
        return nullptr;
    }
    auto* slice = new QuantumSlice();
    slice->num_qubits = num_qubits;
    slice->state_size = 1ULL << num_qubits;
    CUDA_CHECK(cudaMalloc(&slice->device_state, slice->state_size * sizeof(cuDoubleComplex)));
    qs_initialize_basis(slice, 0);
    return slice;
}

void qs_destroy(QuantumSlice* slice) {
    if (!slice) return;
    CUDA_CHECK(cudaFree(slice->device_state));
    delete slice;
}

void qs_initialize_basis(QuantumSlice* slice, int basis_index) {
    if (!slice) return;
    CUDA_CHECK(cudaMemset(slice->device_state, 0, slice->state_size * sizeof(cuDoubleComplex)));
    size_t idx = static_cast<size_t>(basis_index) % slice->state_size;
    cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
    CUDA_CHECK(cudaMemcpy(slice->device_state + idx, &one, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
}

void qs_apply_single_qubit_gate(QuantumSlice* slice, int target_qubit, const cuDoubleComplex* gate_matrix_host) {
    if (!slice || !gate_matrix_host) return;
    cuDoubleComplex gate[4];
    std::memcpy(gate, gate_matrix_host, sizeof(gate));
    dim3 block(256);
    dim3 grid((slice->state_size / 2 + block.x - 1) / block.x);
    kernel_apply_single_qubit_gate<<<grid, block>>>(slice->device_state,
                                                   slice->num_qubits,
                                                   target_qubit,
                                                   gate[0], gate[1], gate[2], gate[3]);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void qs_apply_controlled_phase(QuantumSlice* slice, int control_qubit, int target_qubit, double theta) {
    if (!slice) return;
    dim3 block(256);
    dim3 grid((slice->state_size + block.x - 1) / block.x);
    kernel_apply_controlled_phase<<<grid, block>>>(slice->device_state,
                                                  slice->num_qubits,
                                                  control_qubit,
                                                  target_qubit,
                                                  theta);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void qs_apply_cnot(QuantumSlice* slice, int control_qubit, int target_qubit) {
    if (!slice) return;
    dim3 block(256);
    dim3 grid((slice->state_size + block.x - 1) / block.x);
    kernel_apply_cnot<<<grid, block>>>(slice->device_state,
                                       slice->num_qubits,
                                       control_qubit,
                                       target_qubit);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void qs_inject_depolarizing_noise(QuantumSlice* slice, int target_qubit, double probability, unsigned long long seed) {
    if (!slice) return;
    if (probability <= 0.0) return;
    dim3 block(256);
    dim3 grid((slice->state_size / 2 + block.x - 1) / block.x);
    kernel_depolarizing_noise<<<grid, block>>>(slice->device_state,
                                              slice->num_qubits,
                                              target_qubit,
                                              probability,
                                              seed);
    CUDA_CHECK(cudaDeviceSynchronize());
}

namespace {
__global__ void kernel_coherence(const cuDoubleComplex* state,
                                 double* partial,
                                 size_t size) {
    extern __shared__ double shared[];
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    double accum = 0.0;
    for (size_t i = idx; i < size; i += stride) {
        double mag = cuCabs(state[i]);
        accum += mag * mag;
    }
    shared[threadIdx.x] = accum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            shared[threadIdx.x] += shared[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = shared[0];
    }
}

__global__ void kernel_harmonic_index(const cuDoubleComplex* state,
                                      double* partial,
                                      size_t size) {
    extern __shared__ double shared[];
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    double accum = 0.0;
    for (size_t i = idx; i < size; i += stride) {
        double phase = atan2(state[i].y, state[i].x);
        double cosine = cos(phase);
        accum += cosine;
    }
    shared[threadIdx.x] = accum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            shared[threadIdx.x] += shared[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = shared[0];
    }
}
}

double qs_measure_coherence(const QuantumSlice* slice) {
    if (!slice) return 0.0;
    int blocks = 256;
    int threads = 256;
    std::vector<double> host_partials(blocks);
    double* device_partials = nullptr;
    CUDA_CHECK(cudaMalloc(&device_partials, blocks * sizeof(double)));
    kernel_coherence<<<blocks, threads, threads * sizeof(double)>>>(slice->device_state,
                                                                   device_partials,
                                                                   slice->state_size);
    CUDA_CHECK(cudaMemcpy(host_partials.data(), device_partials, blocks * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device_partials));
    double sum = 0.0;
    for (double v : host_partials) {
        sum += v;
    }
    return sum;
}

double qs_compute_harmonic_index(const QuantumSlice* slice) {
    if (!slice) return 0.0;
    int blocks = 256;
    int threads = 256;
    std::vector<double> host_partials(blocks);
    double* device_partials = nullptr;
    CUDA_CHECK(cudaMalloc(&device_partials, blocks * sizeof(double)));
    kernel_harmonic_index<<<blocks, threads, threads * sizeof(double)>>>(slice->device_state,
                                                                        device_partials,
                                                                        slice->state_size);
    CUDA_CHECK(cudaMemcpy(host_partials.data(), device_partials, blocks * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device_partials));
    double sum = 0.0;
    for (double v : host_partials) {
        sum += v;
    }
    return sum / static_cast<double>(slice->state_size);
}

void qs_checkpoint(const QuantumSlice* slice, const char* path) {
    if (!slice || !path) return;
    cuDoubleComplex* buffer = nullptr;
    CUDA_CHECK(cudaMallocHost(&buffer, slice->state_size * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMemcpy(buffer, slice->device_state, slice->state_size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    std::ofstream ofs(path, std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(&slice->num_qubits), sizeof(int));
    ofs.write(reinterpret_cast<const char*>(buffer), slice->state_size * sizeof(cuDoubleComplex));
    ofs.close();
    CUDA_CHECK(cudaFreeHost(buffer));
}

}  // extern "C"
