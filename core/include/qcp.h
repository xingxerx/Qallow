#ifndef QCP_H
#define QCP_H

#include "qallow_kernel.h"

// Quantum Co-Processor module
// Handles quantum state optimization and entanglement

// QCP Configuration
#define QCP_MAX_QUBITS 64
#define QCP_ENTANGLEMENT_DEPTH 8
#define QCP_OPTIMIZATION_ITERATIONS 10

typedef struct {
    float qubit_states[QCP_MAX_QUBITS * 2]; // Real and imaginary components
    float entanglement_matrix[QCP_MAX_QUBITS][QCP_MAX_QUBITS];
    float optimization_target;
    float current_fidelity;
    int active_qubits;
    bool quantum_advantage_detected;
} qcp_state_t;

// Function declarations
CUDA_CALLABLE void qcp_init(qcp_state_t* qcp);
CUDA_CALLABLE void qcp_process_quantum_layer(qcp_state_t* qcp, overlay_t* overlay);
CUDA_CALLABLE float qcp_calculate_fidelity(const qcp_state_t* qcp);
CUDA_CALLABLE void qcp_apply_optimization_nudge(qcp_state_t* qcp, overlay_t* overlay);
CUDA_CALLABLE void qcp_update_entanglement(qcp_state_t* qcp);

// CUDA-specific QCP functions
#if CUDA_ENABLED
void qcp_cuda_process_qubits(qcp_state_t* qcp, overlay_t* overlays, int num_overlays);
__global__ void qcp_quantum_kernel(float* overlay_data, float* qubit_data, int nodes);
__global__ void qcp_entanglement_kernel(float* entanglement_matrix, int qubits);
#endif

// CPU fallback
void qcp_cpu_process_qubits(qcp_state_t* qcp, overlay_t* overlays, int num_overlays);

#endif // QCP_H