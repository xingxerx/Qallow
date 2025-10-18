#ifndef PPAI_H
#define PPAI_H

#include "qallow_kernel.h"

// Photonic-Probabilistic AI module
// Handles stochastic simulations and photonic emulation

// PPAI Configuration
#define PPAI_MAX_PHOTONS 1024
#define PPAI_WAVELENGTH_COUNT 16
#define PPAI_INTERFERENCE_THRESHOLD 0.85f

typedef struct {
    float photon_intensity[PPAI_MAX_PHOTONS];
    float wavelengths[PPAI_WAVELENGTH_COUNT];
    float coherence_matrix[PPAI_WAVELENGTH_COUNT][PPAI_WAVELENGTH_COUNT];
    float quantum_noise_level;
    bool photonic_mode_active;
} ppai_state_t;

// Function declarations
CUDA_CALLABLE void ppai_init(ppai_state_t* ppai);
CUDA_CALLABLE void ppai_process_photonic_layer(ppai_state_t* ppai, overlay_t* overlay);
CUDA_CALLABLE float ppai_calculate_interference(const ppai_state_t* ppai, int wavelength_idx);
CUDA_CALLABLE void ppai_apply_quantum_noise(ppai_state_t* ppai, float noise_factor);

// CUDA-specific PPAI functions
#if CUDA_ENABLED
void ppai_cuda_process_photons(ppai_state_t* ppai, overlay_t* overlays, int num_overlays);
__global__ void ppai_photonic_kernel(float* overlay_data, float* photon_data, int nodes);
#endif

// CPU fallback
void ppai_cpu_process_photons(ppai_state_t* ppai, overlay_t* overlays, int num_overlays);

#endif // PPAI_H