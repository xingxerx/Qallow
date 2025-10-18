#include "qcp.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Quantum Co-Processor module - CPU implementation

CUDA_CALLABLE void qcp_init(qcp_state_t* qcp) {
    if (!qcp) return;
    
    // Initialize qubits in superposition state
    for (int i = 0; i < QCP_MAX_QUBITS; i++) {
        float angle = (float)i * 2.0f * 3.14159f / QCP_MAX_QUBITS;
        qcp->qubit_states[i * 2] = cosf(angle) / sqrtf(2.0f);     // Real part
        qcp->qubit_states[i * 2 + 1] = sinf(angle) / sqrtf(2.0f); // Imaginary part
    }
    
    // Initialize entanglement matrix
    for (int i = 0; i < QCP_MAX_QUBITS; i++) {
        for (int j = 0; j < QCP_MAX_QUBITS; j++) {
            if (i == j) {
                qcp->entanglement_matrix[i][j] = 1.0f;
            } else {
                float distance = fabsf((float)(i - j));
                qcp->entanglement_matrix[i][j] = expf(-distance / 10.0f);
            }
        }
    }
    
    qcp->optimization_target = 0.98f;
    qcp->current_fidelity = 0.5f;
    qcp->active_qubits = QCP_MAX_QUBITS;
    qcp->quantum_advantage_detected = false;
    
    printf("[QCP] Quantum Co-Processor module initialized\n");
}

CUDA_CALLABLE void qcp_process_quantum_layer(qcp_state_t* qcp, overlay_t* overlay) {
    if (!qcp || !overlay) return;
    
    qcp_apply_optimization_nudge(qcp, overlay);
    qcp->current_fidelity = qcp_calculate_fidelity(qcp);
    
    if (qcp->current_fidelity > qcp->optimization_target) {
        qcp->quantum_advantage_detected = true;
    }
}

CUDA_CALLABLE float qcp_calculate_fidelity(const qcp_state_t* qcp) {
    if (!qcp) return 0.0f;
    
    float total_fidelity = 0.0f;
    
    for (int i = 0; i < qcp->active_qubits; i++) {
        float real_part = qcp->qubit_states[i * 2];
        float imag_part = qcp->qubit_states[i * 2 + 1];
        float state_purity = real_part * real_part + imag_part * imag_part;
        total_fidelity += state_purity;
    }
    
    return total_fidelity / qcp->active_qubits;
}

CUDA_CALLABLE void qcp_apply_optimization_nudge(qcp_state_t* qcp, overlay_t* overlay) {
    if (!qcp || !overlay) return;
    
    for (int node = 0; node < overlay->node_count && node < qcp->active_qubits; node++) {
        // Calculate optimization direction
        float target_value = qcp->optimization_target;
        float current_value = overlay->values[node];
        float nudge = (target_value - current_value) * 0.01f;
        
        // Apply clamped nudge
        if (nudge < -0.05f) nudge = -0.05f;
        if (nudge > 0.05f) nudge = 0.05f;
        
        overlay->values[node] += nudge;
        
        // Clamp to valid range
        if (overlay->values[node] < 0.0f) overlay->values[node] = 0.0f;
        if (overlay->values[node] > 1.0f) overlay->values[node] = 1.0f;
    }
}

CUDA_CALLABLE void qcp_update_entanglement(qcp_state_t* qcp) {
    if (!qcp) return;
    
    for (int i = 0; i < qcp->active_qubits; i++) {
        for (int j = i + 1; j < qcp->active_qubits; j++) {
            // Calculate entanglement based on qubit state correlation
            float corr_real = qcp->qubit_states[i * 2] * qcp->qubit_states[j * 2];
            float corr_imag = qcp->qubit_states[i * 2 + 1] * qcp->qubit_states[j * 2 + 1];
            float correlation = sqrtf(corr_real * corr_real + corr_imag * corr_imag);
            
            qcp->entanglement_matrix[i][j] = correlation;
            qcp->entanglement_matrix[j][i] = correlation; // Symmetric matrix
        }
    }
}

// CPU fallback implementation
void qcp_cpu_process_qubits(qcp_state_t* qcp, overlay_t* overlays, int num_overlays) {
    if (!qcp || !overlays) return;
    
    // Update entanglement matrix
    qcp_update_entanglement(qcp);
    
    for (int overlay_idx = 0; overlay_idx < num_overlays; overlay_idx++) {
        overlay_t* overlay = &overlays[overlay_idx];
        
        for (int node = 0; node < overlay->node_count && node < QCP_MAX_QUBITS; node++) {
            // Get qubit state
            float real_part = qcp->qubit_states[node * 2];
            float imag_part = qcp->qubit_states[node * 2 + 1];
            
            // Calculate probability
            float probability = real_part * real_part + imag_part * imag_part;
            
            // Apply quantum optimization
            float optimization_factor = 0.02f * (probability - 0.5f);
            overlay->values[node] += optimization_factor;
            
            // Ensure valid range
            if (overlay->values[node] < 0.0f) overlay->values[node] = 0.0f;
            if (overlay->values[node] > 1.0f) overlay->values[node] = 1.0f;
        }
        
        // Update stability
        overlay->stability = qallow_calculate_stability(overlay);
    }
}

