#ifndef OVERLAY_H
#define OVERLAY_H

#include "qallow_kernel.h"

// Overlay data structures and management
// Handles the three-layer overlay system: Orbital, River-Delta, Mycelial

// Overlay interaction patterns
#define OVERLAY_INTERACTION_STRENGTH 0.15f
#define OVERLAY_DAMPING_FACTOR 0.02f
#define OVERLAY_RESONANCE_FREQUENCY 0.1f

// Overlay-specific constants
#define ORBITAL_ROTATION_SPEED 0.05f
#define RIVER_DELTA_FLOW_RATE 0.03f
#define MYCELIAL_GROWTH_RATE 0.01f

typedef struct {
    float x, y, z;
} vector3_t;

typedef struct {
    vector3_t position;
    float energy_level;
    float phase;
    bool active;
} overlay_node_t;

typedef struct {
    overlay_node_t nodes[MAX_NODES];
    float interaction_matrix[MAX_NODES][MAX_NODES];
    float resonance_frequency;
    float damping_coefficient;
    overlay_type_t type;
    
    // Overlay-specific parameters
    union {
        struct { // Orbital overlay
            float orbital_radius;
            float angular_velocity;
            float inclination;
        } orbital;
        
        struct { // River-Delta overlay
            float flow_velocity;
            float turbulence_level;
            float sediment_density;
        } river_delta;
        
        struct { // Mycelial overlay
            float growth_rate;
            float nutrient_density;
            float network_connectivity;
        } mycelial;
    } params;
} overlay_extended_t;

// Function declarations
CUDA_CALLABLE void overlay_init(overlay_t* overlay, overlay_type_t type, int node_count);
CUDA_CALLABLE void overlay_update(overlay_t* overlay, overlay_type_t type);
CUDA_CALLABLE void overlay_apply_interactions(overlay_t* overlays, int num_overlays);
CUDA_CALLABLE float overlay_calculate_resonance(const overlay_t* overlay1, const overlay_t* overlay2);

// Extended overlay functions
CUDA_CALLABLE void overlay_extended_init(overlay_extended_t* overlay, overlay_type_t type);
CUDA_CALLABLE void overlay_extended_update(overlay_extended_t* overlay);
CUDA_CALLABLE void overlay_calculate_interactions(overlay_extended_t* overlays, int count);

// CUDA-specific overlay functions
#if CUDA_ENABLED
void overlay_cuda_process_all(overlay_t* overlays, int num_overlays, int nodes);
__global__ void overlay_update_kernel(float* overlay_data, int nodes, int overlay_type);
__global__ void overlay_interaction_kernel(float* overlay1, float* overlay2, int nodes);
#endif

// CPU fallback
void overlay_cpu_process_all(overlay_t* overlays, int num_overlays, int nodes);

// Utility functions
void overlay_print_status(const overlay_t* overlay, overlay_type_t type);
const char* overlay_type_name(overlay_type_t type);

#endif // OVERLAY_H