/*
 * Cirq Quantum Processor - C Bridge
 * Python C API interface to Google Cirq quantum simulation framework
 * Provides real quantum circuit simulation using Cirq backend
 */

#ifndef CIRQ_QUANTUM_H
#define CIRQ_QUANTUM_H

#include <stdint.h>
#include <stdbool.h>
#include <Python.h>

/* Quantum Circuit State */
typedef struct {
    uint32_t num_qubits;
    uint32_t num_operations;
    bool initialized;
    char circuit_name[256];
} quantum_circuit_t;

/* Quantum Measurement Result */
typedef struct {
    uint32_t num_qubits;
    uint32_t num_shots;
    uint8_t *measurements;  /* Bitwise measurement results */
    double *probabilities;  /* Probability distribution */
    uint64_t total_counts;
} quantum_result_t;

/* Quantum Processor Context */
typedef struct {
    PyObject *cirq_module;
    PyObject *simulator;
    bool initialized;
    char backend_name[128];
    
    /* Statistics */
    uint64_t circuits_run;
    uint64_t total_shots;
    double total_simulation_time_ms;
} quantum_context_t;

/* Function Declarations */

/**
 * Initialize Cirq quantum processor
 */
quantum_context_t* quantum_init(void);

/**
 * Cleanup quantum context
 */
void quantum_cleanup(quantum_context_t *ctx);

/**
 * Create quantum circuit
 */
quantum_circuit_t* quantum_create_circuit(quantum_context_t *ctx, uint32_t num_qubits,
                                         const char *name);

/**
 * Destroy quantum circuit
 */
void quantum_destroy_circuit(quantum_circuit_t *circuit);

/**
 * Add Hadamard gate
 */
bool quantum_add_h_gate(quantum_context_t *ctx, quantum_circuit_t *circuit,
                       uint32_t target_qubit);

/**
 * Add X (Pauli X) gate
 */
bool quantum_add_x_gate(quantum_context_t *ctx, quantum_circuit_t *circuit,
                       uint32_t target_qubit);

/**
 * Add Y (Pauli Y) gate
 */
bool quantum_add_y_gate(quantum_context_t *ctx, quantum_circuit_t *circuit,
                       uint32_t target_qubit);

/**
 * Add Z (Pauli Z) gate
 */
bool quantum_add_z_gate(quantum_context_t *ctx, quantum_circuit_t *circuit,
                       uint32_t target_qubit);

/**
 * Add CNOT (CX) gate
 */
bool quantum_add_cnot_gate(quantum_context_t *ctx, quantum_circuit_t *circuit,
                          uint32_t control_qubit, uint32_t target_qubit);

/**
 * Add Rx rotation gate (angle in radians)
 */
bool quantum_add_rx_gate(quantum_context_t *ctx, quantum_circuit_t *circuit,
                        uint32_t target_qubit, double angle_rad);

/**
 * Add Rz rotation gate (angle in radians)
 */
bool quantum_add_rz_gate(quantum_context_t *ctx, quantum_circuit_t *circuit,
                        uint32_t target_qubit, double angle_rad);

/**
 * Add measurement
 */
bool quantum_add_measurement(quantum_context_t *ctx, quantum_circuit_t *circuit,
                            uint32_t qubit, const char *measurement_key);

/**
 * Run quantum circuit simulation
 */
quantum_result_t* quantum_run_circuit(quantum_context_t *ctx, quantum_circuit_t *circuit,
                                      uint32_t num_shots);

/**
 * Destroy quantum result
 */
void quantum_destroy_result(quantum_result_t *result);

/**
 * Get measurement statistics
 */
double quantum_get_probability(quantum_result_t *result, uint32_t state);

/**
 * Print quantum result
 */
void quantum_print_result(quantum_result_t *result);

/**
 * Check quantum processor status
 */
void quantum_print_status(quantum_context_t *ctx);

/**
 * Check for Cirq availability
 */
bool quantum_is_available(void);

#endif /* CIRQ_QUANTUM_H */
