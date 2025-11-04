#include "qcp.h"




#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#if CUDA_ENABLED
#include <curand_kernel.h>


__global__ void qcp_quantum_kernel(float* overlay_data, float* qubit_data, int nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < nodes && idx < QCP_MAX_QUBITS) {

        float real_part = qubit_data[idx * 2];
        float imag_part = qubit_data[idx * 2 + 1];
        

        float probability = real_part * real_part + imag_part * imag_part;
        

        float optimization_factor = 0.02f * (probability - 0.5f);
        overlay_data[idx] += optimization_factor;
        

        overlay_data[idx] = fmaxf(0.0f, fminf(1.0f, overlay_data[idx]));
    }
}


__global__ void qcp_entanglement_kernel(float* entanglement_matrix, int qubits) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < qubits && j < qubits && i != j) {

        float distance = sqrtf((float)((i - j) * (i - j)));
        float entanglement_strength = expf(-distance / 10.0f);
        

        entanglement_matrix[i * qubits + j] = entanglement_strength;
    }
}


void qcp_cuda_process_qubits(qcp_state_t* qcp, overlay_t* overlays, int num_overlays) {
    static float* d_overlay_data = nullptr;
    static float* d_qubit_data = nullptr;
    static bool cuda_initialized = false;
    
    if (!cuda_initialized) {

        cudaMalloc(&d_overlay_data, MAX_NODES * sizeof(float));
        cudaMalloc(&d_qubit_data, QCP_MAX_QUBITS * 2 * sizeof(float));
        

        cudaMemcpy(d_qubit_data, qcp->qubit_states, 
                   QCP_MAX_QUBITS * 2 * sizeof(float), cudaMemcpyHostToDevice);
        
        cuda_initialized = true;
    }
    

    for (int overlay_idx = 0; overlay_idx < num_overlays; overlay_idx++) {
        overlay_t* overlay = &overlays[overlay_idx];
        

        cudaMemcpy(d_overlay_data, overlay->values, 
                   overlay->node_count * sizeof(float), cudaMemcpyHostToDevice);
        

        int threads_per_block = 256;
        int blocks = (overlay->node_count + threads_per_block - 1) / threads_per_block;
        qcp_quantum_kernel<<<blocks, threads_per_block>>>(d_overlay_data, d_qubit_data, overlay->node_count);
        

        cudaMemcpy(overlay->values, d_overlay_data, 
                   overlay->node_count * sizeof(float), cudaMemcpyDeviceToHost);
        

        overlay->stability = qallow_calculate_stability(overlay);
    }
    
    cudaDeviceSynchronize();
}

#else


void qcp_cuda_process_qubits(qcp_state_t* qcp, overlay_t* overlays, int num_overlays) {

}

#endif // CUDA_ENABLED

