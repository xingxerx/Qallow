#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cstddef>

extern "C" {

typedef struct {
    cuDoubleComplex* device_state;
    int num_qubits;
    size_t state_size;
} QuantumSlice;

QuantumSlice* qs_create(int num_qubits);
void qs_destroy(QuantumSlice* slice);
void qs_initialize_basis(QuantumSlice* slice, int basis_index);
void qs_apply_single_qubit_gate(QuantumSlice* slice, int target_qubit, const cuDoubleComplex* gate_matrix_host);
void qs_apply_controlled_phase(QuantumSlice* slice, int control_qubit, int target_qubit, double theta);
void qs_apply_cnot(QuantumSlice* slice, int control_qubit, int target_qubit);
void qs_inject_depolarizing_noise(QuantumSlice* slice, int target_qubit, double probability, unsigned long long seed);
double qs_measure_coherence(const QuantumSlice* slice);
double qs_compute_harmonic_index(const QuantumSlice* slice);
void qs_checkpoint(const QuantumSlice* slice, const char* path);

}
