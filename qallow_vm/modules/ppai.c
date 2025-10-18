#include "ppai.h"
#include <math.h>

// Photonic-Probabilistic AI module - CPU implementation

CUDA_CALLABLE void ppai_init(ppai_state_t* ppai) {
    if (!ppai) return;
    
    // Initialize photon intensities
    for (int i = 0; i < PPAI_MAX_PHOTONS; i++) {
        ppai->photon_intensity[i] = 0.5f + ((float)rand() / RAND_MAX) * 0.5f;
    }
    
    // Initialize wavelengths (visible spectrum simulation)
    for (int i = 0; i < PPAI_WAVELENGTH_COUNT; i++) {
        ppai->wavelengths[i] = 380.0f + (float)i * (780.0f - 380.0f) / PPAI_WAVELENGTH_COUNT;
    }
    
    // Initialize coherence matrix
    for (int i = 0; i < PPAI_WAVELENGTH_COUNT; i++) {
        for (int j = 0; j < PPAI_WAVELENGTH_COUNT; j++) {
            if (i == j) {
                ppai->coherence_matrix[i][j] = 1.0f;
            } else {
                float wavelength_diff = fabsf(ppai->wavelengths[i] - ppai->wavelengths[j]);
                ppai->coherence_matrix[i][j] = expf(-wavelength_diff / 100.0f);
            }
        }
    }
    
    ppai->quantum_noise_level = 0.01f;
    ppai->photonic_mode_active = true;
    
    printf("[PPAI] Photonic-Probabilistic AI module initialized\n");
}

CUDA_CALLABLE void ppai_process_photonic_layer(ppai_state_t* ppai, overlay_t* overlay) {
    if (!ppai || !overlay || !ppai->photonic_mode_active) return;
    
    for (int node = 0; node < overlay->node_count; node++) {
        float interference = ppai_calculate_interference(ppai, node % PPAI_WAVELENGTH_COUNT);
        overlay->values[node] *= (1.0f + interference * 0.1f);
        
        // Clamp to valid range
        if (overlay->values[node] < 0.0f) overlay->values[node] = 0.0f;
        if (overlay->values[node] > 1.0f) overlay->values[node] = 1.0f;
    }
}

CUDA_CALLABLE float ppai_calculate_interference(const ppai_state_t* ppai, int wavelength_idx) {
    if (!ppai || wavelength_idx < 0 || wavelength_idx >= PPAI_WAVELENGTH_COUNT) return 0.0f;
    
    float total_interference = 0.0f;
    
    for (int i = 0; i < PPAI_WAVELENGTH_COUNT; i++) {
        total_interference += ppai->coherence_matrix[wavelength_idx][i] * 
                             sinf(ppai->wavelengths[i] * 0.01f);
    }
    
    return total_interference / PPAI_WAVELENGTH_COUNT;
}

CUDA_CALLABLE void ppai_apply_quantum_noise(ppai_state_t* ppai, float noise_factor) {
    if (!ppai) return;
    
    if (noise_factor < 0.001f) noise_factor = 0.001f;
    if (noise_factor > 0.1f) noise_factor = 0.1f;
    
    ppai->quantum_noise_level = noise_factor;
}

// CPU fallback implementation
void ppai_cpu_process_photons(ppai_state_t* ppai, overlay_t* overlays, int num_overlays) {
    if (!ppai || !overlays) return;
    
    for (int overlay_idx = 0; overlay_idx < num_overlays; overlay_idx++) {
        overlay_t* overlay = &overlays[overlay_idx];
        
        for (int node = 0; node < overlay->node_count; node++) {
            // Simulate photonic interference
            float photon_intensity = ppai->photon_intensity[node % PPAI_MAX_PHOTONS];
            float interference_factor = sinf(photon_intensity * 2.0f * 3.14159f) * 0.1f;
            
            // Add quantum noise
            float quantum_noise = ((float)rand() / RAND_MAX - 0.5f) * ppai->quantum_noise_level;
            
            // Apply effects
            overlay->values[node] += interference_factor + quantum_noise;
            
            // Clamp to valid range
            if (overlay->values[node] < 0.0f) overlay->values[node] = 0.0f;
            if (overlay->values[node] > 1.0f) overlay->values[node] = 1.0f;
        }
        
        // Update stability
        overlay->stability = qallow_calculate_stability(overlay);
    }
}

