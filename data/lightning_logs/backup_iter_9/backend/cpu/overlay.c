#include "overlay.h"
#include <string.h>

// Overlay management - Three-layer system (Orbital, River-Delta, Mycelial)

void overlay_init(overlay_t* overlay, overlay_type_t type, int node_count) {
    if (!overlay) return;
    (void)type;
    
    memset(overlay, 0, sizeof(overlay_t));
    
    overlay->node_count = node_count > MAX_NODES ? MAX_NODES : node_count;
    overlay->stability = 0.5f;
    
    // Initialize node values
    for (int i = 0; i < overlay->node_count; i++) {
        overlay->values[i] = 0.5f + (float)rand() / RAND_MAX * 0.1f;
        overlay->history[i] = overlay->values[i];
    }
}

CUDA_CALLABLE void overlay_update(overlay_t* overlay, overlay_type_t type) {
    if (!overlay) return;
    
    // Type-specific update logic
    switch (type) {
        case OVERLAY_ORBITAL:
            // Orbital: rotational dynamics
            for (int i = 0; i < overlay->node_count; i++) {
                overlay->values[i] += ORBITAL_ROTATION_SPEED * 0.01f;
                if (overlay->values[i] > 1.0f) overlay->values[i] -= 1.0f;
            }
            break;
            
        case OVERLAY_RIVER_DELTA:
            // River-Delta: flow dynamics
            for (int i = 0; i < overlay->node_count; i++) {
                overlay->values[i] += RIVER_DELTA_FLOW_RATE * 0.01f;
                if (overlay->values[i] > 1.0f) overlay->values[i] = 1.0f;
                if (overlay->values[i] < 0.0f) overlay->values[i] = 0.0f;
            }
            break;
            
        case OVERLAY_MYCELIAL:
            // Mycelial: growth dynamics
            for (int i = 0; i < overlay->node_count; i++) {
                overlay->values[i] += MYCELIAL_GROWTH_RATE * 0.01f;
                if (overlay->values[i] > 1.0f) overlay->values[i] = 1.0f;
            }
            break;
    }
}

CUDA_CALLABLE void overlay_apply_interactions(overlay_t* overlays, int num_overlays) {
    if (!overlays || num_overlays < 2) return;
    
    // Apply cross-overlay interactions
    for (int i = 0; i < num_overlays; i++) {
        for (int j = i + 1; j < num_overlays; j++) {
            overlay_t* o1 = &overlays[i];
            overlay_t* o2 = &overlays[j];
            
            // Simple interaction: average nearby nodes
            for (int k = 0; k < o1->node_count && k < o2->node_count; k++) {
                float interaction = (o1->values[k] - o2->values[k]) * OVERLAY_INTERACTION_STRENGTH;
                o1->values[k] -= interaction;
                o2->values[k] += interaction;
                
                // Clamp
                if (o1->values[k] < 0.0f) o1->values[k] = 0.0f;
                if (o1->values[k] > 1.0f) o1->values[k] = 1.0f;
                if (o2->values[k] < 0.0f) o2->values[k] = 0.0f;
                if (o2->values[k] > 1.0f) o2->values[k] = 1.0f;
            }
        }
    }
}

CUDA_CALLABLE float overlay_calculate_resonance(const overlay_t* overlay1, const overlay_t* overlay2) {
    if (!overlay1 || !overlay2) return 0.0f;
    
    float correlation = 0.0f;
    int count = overlay1->node_count < overlay2->node_count ? overlay1->node_count : overlay2->node_count;
    
    for (int i = 0; i < count; i++) {
        correlation += overlay1->values[i] * overlay2->values[i];
    }
    
    return correlation / count;
}

CUDA_CALLABLE void overlay_extended_init(overlay_extended_t* overlay, overlay_type_t type) {
    if (!overlay) return;
    
    memset(overlay, 0, sizeof(overlay_extended_t));
    overlay->type = type;
    
    // Initialize nodes
    for (int i = 0; i < MAX_NODES; i++) {
        overlay->nodes[i].position.x = (float)rand() / RAND_MAX;
        overlay->nodes[i].position.y = (float)rand() / RAND_MAX;
        overlay->nodes[i].position.z = (float)rand() / RAND_MAX;
        overlay->nodes[i].energy_level = 0.5f;
        overlay->nodes[i].phase = 0.0f;
        overlay->nodes[i].active = true;
    }
}

CUDA_CALLABLE void overlay_extended_update(overlay_extended_t* overlay) {
    if (!overlay) return;
    
    // Update node states
    for (int i = 0; i < MAX_NODES; i++) {
        if (!overlay->nodes[i].active) continue;
        
        // Update phase
        overlay->nodes[i].phase += OVERLAY_RESONANCE_FREQUENCY;
        if (overlay->nodes[i].phase > 2.0f * 3.14159f) {
            overlay->nodes[i].phase -= 2.0f * 3.14159f;
        }
        
        // Update energy with damping
        overlay->nodes[i].energy_level *= (1.0f - OVERLAY_DAMPING_FACTOR);
    }
}

CUDA_CALLABLE void overlay_calculate_interactions(overlay_extended_t* overlays, int count) {
    if (!overlays || count < 2) return;
    
    for (int i = 0; i < count; i++) {
        for (int j = i + 1; j < count; j++) {
            // Calculate interaction between overlays
            float interaction = 0.0f;
            for (int k = 0; k < MAX_NODES; k++) {
                if (overlays[i].nodes[k].active && overlays[j].nodes[k].active) {
                    interaction += overlays[i].nodes[k].energy_level * overlays[j].nodes[k].energy_level;
                }
            }
            // Store in interaction matrix (simplified)
            overlays[i].interaction_matrix[0][0] = interaction;
        }
    }
}

void overlay_cpu_process_all(overlay_t* overlays, int num_overlays, int nodes) {
    if (!overlays) return;
    (void)nodes;
    
    for (int i = 0; i < num_overlays; i++) {
        overlay_update(&overlays[i], (overlay_type_t)i);
    }
    
    overlay_apply_interactions(overlays, num_overlays);
}

void overlay_print_status(const overlay_t* overlay, overlay_type_t type) {
    if (!overlay) return;
    
    printf("[%s] Stability: %.4f | Nodes: %d\n",
           overlay_type_name(type), overlay->stability, overlay->node_count);
}

const char* overlay_type_name(overlay_type_t type) {
    switch (type) {
        case OVERLAY_ORBITAL: return "ORBITAL";
        case OVERLAY_RIVER_DELTA: return "RIVER_DELTA";
        case OVERLAY_MYCELIAL: return "MYCELIAL";
        default: return "UNKNOWN";
    }
}

#if CUDA_ENABLED
void overlay_cuda_process_all(overlay_t* overlays, int num_overlays, int nodes) {
    if (!overlays) return;
    (void)num_overlays;
    (void)nodes;
    // CUDA processing stub
}
#endif
