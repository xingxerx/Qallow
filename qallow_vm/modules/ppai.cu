#include "ppai.h"




#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#if CUDA_ENABLED
#include <curand_kernel.h>


__global__ void ppai_photonic_kernel(float* overlay_data, float* photon_data, int nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < nodes) {

        curandState_t state;
        curand_init(clock64(), idx, 0, &state);
        

        float photon_intensity = photon_data[idx % PPAI_MAX_PHOTONS];
        float quantum_noise = curand_normal(&state) * 0.01f;
        

        float interference_factor = sinf(photon_intensity * 2.0f * M_PI) * 0.1f;
        overlay_data[idx] += interference_factor + quantum_noise;
        

        overlay_data[idx] = fmaxf(0.0f, fminf(1.0f, overlay_data[idx]));
    }
}


void ppai_cuda_process_photons(ppai_state_t* ppai, overlay_t* overlays, int num_overlays) {
    static float* d_overlay_data = nullptr;
    static float* d_photon_data = nullptr;
    static bool cuda_initialized = false;
    
    if (!cuda_initialized) {

        cudaMalloc(&d_overlay_data, MAX_NODES * sizeof(float));
        cudaMalloc(&d_photon_data, PPAI_MAX_PHOTONS * sizeof(float));
        

        cudaMemcpy(d_photon_data, ppai->photon_intensity, 
                   PPAI_MAX_PHOTONS * sizeof(float), cudaMemcpyHostToDevice);
        
        cuda_initialized = true;
    }
    

    for (int overlay_idx = 0; overlay_idx < num_overlays; overlay_idx++) {
        overlay_t* overlay = &overlays[overlay_idx];
        

        cudaMemcpy(d_overlay_data, overlay->values, 
                   overlay->node_count * sizeof(float), cudaMemcpyHostToDevice);
        

        int threads_per_block = 256;
        int blocks = (overlay->node_count + threads_per_block - 1) / threads_per_block;
        ppai_photonic_kernel<<<blocks, threads_per_block>>>(d_overlay_data, d_photon_data, overlay->node_count);
        

        cudaMemcpy(overlay->values, d_overlay_data, 
                   overlay->node_count * sizeof(float), cudaMemcpyDeviceToHost);
        

        overlay->stability = qallow_calculate_stability(overlay);
    }
    
    cudaDeviceSynchronize();
}

#else


void ppai_cuda_process_photons(ppai_state_t* ppai, overlay_t* overlays, int num_overlays) {

}

#endif // CUDA_ENABLED

