/*
 * CUDA-Q Quick Start Examples for Qallow
 * Demonstrates basic quantum circuits and integration with Qallow
 * Converted from Python to C for native integration
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* CUDA-Q C API headers */
#include "cudaq.h"

/* ========================================================================== */
/* Example 1: Bell State (Entanglement)                                      */
/* ========================================================================== */
void example_bell_state() {
    printf("\n%s\n", "========================================================================");
    printf("Example 1: Bell State (Entanglement)\n");
    printf("%s\n", "========================================================================");
    
    /* Create quantum kernel for Bell state */
    cudaq_kernel kernel = cudaq_kernel_create("bell_state");
    
    /* Allocate 2 qubits */
    cudaq_qvector qubits = cudaq_qvector_create(2);
    
    /* Apply Hadamard to first qubit */
    cudaq_h(qubits, 0);
    
    /* Apply CNOT (controlled-X) */
    cudaq_cx(qubits, 0, 1);
    
    /* Measure all qubits */
    cudaq_mz(qubits);
    
    /* Sample the circuit */
    cudaq_sample_result result = cudaq_sample(kernel, 1000);
    
    printf("\nBell State Results (1000 shots):\n");
    printf("Expected: ~500 '00' and ~500 '11' (maximally entangled)\n");
    
    /* Print results */
    for (int i = 0; i < result.num_states; i++) {
        printf("  %s: %d\n", result.states[i], result.counts[i]);
    }
    
    /* Cleanup */
    cudaq_sample_result_free(&result);
    cudaq_qvector_free(&qubits);
    cudaq_kernel_free(&kernel);
}

/* ========================================================================== */
/* Example 2: Superposition                                                  */
/* ========================================================================== */
void example_superposition() {
    printf("\n%s\n", "========================================================================");
    printf("Example 2: Superposition\n");
    printf("%s\n", "========================================================================");
    
    cudaq_kernel kernel = cudaq_kernel_create("superposition");
    cudaq_qvector qubits = cudaq_qvector_create(3);
    
    /* Apply Hadamard to all qubits */
    for (int i = 0; i < 3; i++) {
        cudaq_h(qubits, i);
    }
    
    /* Measure all qubits */
    cudaq_mz(qubits);
    
    /* Sample the circuit */
    cudaq_sample_result result = cudaq_sample(kernel, 1000);
    
    printf("\nSuperposition Results (1000 shots):\n");
    printf("Expected: ~125 counts for each of 8 possible states\n");
    
    for (int i = 0; i < result.num_states; i++) {
        printf("  %s: %d\n", result.states[i], result.counts[i]);
    }
    
    cudaq_sample_result_free(&result);
    cudaq_qvector_free(&qubits);
    cudaq_kernel_free(&kernel);
}

/* ========================================================================== */
/* Example 3: Quantum Phase Estimation                                       */
/* ========================================================================== */
void example_phase_estimation() {
    printf("\n%s\n", "========================================================================");
    printf("Example 3: Quantum Phase Estimation\n");
    printf("%s\n", "========================================================================");
    
    double angle = M_PI / 2;  /* π/2 */
    
    cudaq_kernel kernel = cudaq_kernel_create("phase_estimation");
    cudaq_qvector qubits = cudaq_qvector_create(2);
    
    /* Apply Hadamard to first qubit */
    cudaq_h(qubits, 0);
    
    /* Apply RZ rotation to second qubit */
    cudaq_rz(angle, qubits, 1);
    
    /* Apply CNOT */
    cudaq_cx(qubits, 0, 1);
    
    /* Apply Hadamard to first qubit */
    cudaq_h(qubits, 0);
    
    /* Measure */
    cudaq_mz(qubits);
    
    /* Sample */
    cudaq_sample_result result = cudaq_sample(kernel, 100);
    
    printf("\nPhase Estimation Results (angle=%.3f):\n", angle);
    for (int i = 0; i < result.num_states; i++) {
        printf("  %s: %d\n", result.states[i], result.counts[i]);
    }
    
    cudaq_sample_result_free(&result);
    cudaq_qvector_free(&qubits);
    cudaq_kernel_free(&kernel);
}

/* ========================================================================== */
/* Example 4: Grover's Algorithm (2-qubit)                                   */
/* ========================================================================== */
void example_grovers_algorithm() {
    printf("\n%s\n", "========================================================================");
    printf("Example 4: Grover's Algorithm\n");
    printf("%s\n", "========================================================================");
    
    cudaq_kernel kernel = cudaq_kernel_create("grovers_algorithm");
    cudaq_qvector qubits = cudaq_qvector_create(2);
    
    /* Initialize superposition */
    for (int i = 0; i < 2; i++) {
        cudaq_h(qubits, i);
    }
    
    /* Oracle: mark |11⟩ */
    cudaq_z(qubits, 0);
    cudaq_z(qubits, 1);
    cudaq_cx(qubits, 0, 1);
    cudaq_z(qubits, 0);
    cudaq_z(qubits, 1);
    
    /* Diffusion operator */
    for (int i = 0; i < 2; i++) {
        cudaq_h(qubits, i);
    }
    for (int i = 0; i < 2; i++) {
        cudaq_x(qubits, i);
    }
    cudaq_cx(qubits, 0, 1);
    for (int i = 0; i < 2; i++) {
        cudaq_x(qubits, i);
    }
    for (int i = 0; i < 2; i++) {
        cudaq_h(qubits, i);
    }
    
    /* Measure */
    cudaq_mz(qubits);
    
    /* Sample */
    cudaq_sample_result result = cudaq_sample(kernel, 1000);
    
    printf("\nGrover's Algorithm Results (1000 shots):\n");
    printf("Expected: High probability for |11⟩ (marked state)\n");
    
    for (int i = 0; i < result.num_states; i++) {
        printf("  %s: %d\n", result.states[i], result.counts[i]);
    }
    
    cudaq_sample_result_free(&result);
    cudaq_qvector_free(&qubits);
    cudaq_kernel_free(&kernel);
}

/* ========================================================================== */
/* Example 5: Available Targets                                              */
/* ========================================================================== */
void example_available_targets() {
    printf("\n%s\n", "========================================================================");
    printf("Example 5: Available Quantum Backends\n");
    printf("%s\n", "========================================================================");
    
    /* Get available targets */
    const char** targets = cudaq_get_targets();
    int num_targets = 0;
    
    printf("\nAvailable CUDA-Q targets:\n");
    for (int i = 0; targets[i] != NULL; i++) {
        printf("  • %s\n", targets[i]);
        num_targets++;
    }
    
    /* Get current target */
    const char* current = cudaq_get_target();
    printf("\nCurrent target: %s\n", current);
    
    free(targets);
}

/* ========================================================================== */
/* Example 6: Parameterized Circuit                                          */
/* ========================================================================== */
void example_parameterized_circuit() {
    printf("\n%s\n", "========================================================================");
    printf("Example 6: Parameterized Circuit\n");
    printf("%s\n", "========================================================================");
    
    double theta = M_PI / 4;
    double phi = M_PI / 3;
    
    cudaq_kernel kernel = cudaq_kernel_create("parameterized_circuit");
    cudaq_qvector qubits = cudaq_qvector_create(2);
    
    /* Apply RY rotation to first qubit */
    cudaq_ry(theta, qubits, 0);
    
    /* Apply RZ rotation to second qubit */
    cudaq_rz(phi, qubits, 1);
    
    /* Apply CNOT */
    cudaq_cx(qubits, 0, 1);
    
    /* Measure */
    cudaq_mz(qubits);
    
    /* Sample */
    cudaq_sample_result result = cudaq_sample(kernel, 100);
    
    printf("\nParameterized Circuit Results (θ=%.3f, φ=%.3f):\n", theta, phi);
    for (int i = 0; i < result.num_states; i++) {
        printf("  %s: %d\n", result.states[i], result.counts[i]);
    }
    
    cudaq_sample_result_free(&result);
    cudaq_qvector_free(&qubits);
    cudaq_kernel_free(&kernel);
}

/* ========================================================================== */
/* Main Entry Point                                                           */
/* ========================================================================== */
int main(int argc, char* argv[]) {
    printf("\n%s\n", "========================================================================");
    printf("CUDA-Q Quick Start Examples for Qallow (C Version)\n");
    printf("%s\n", "========================================================================");
    
    /* Initialize CUDA-Q */
    if (cudaq_init() != CUDAQ_SUCCESS) {
        fprintf(stderr, "❌ Failed to initialize CUDA-Q\n");
        return 1;
    }
    
    printf("✅ CUDA-Q initialized successfully!\n");
    
    /* Run examples */
    example_bell_state();
    example_superposition();
    example_phase_estimation();
    example_grovers_algorithm();
    example_available_targets();
    example_parameterized_circuit();
    
    /* Cleanup */
    cudaq_finalize();
    
    /* Summary */
    printf("\n%s\n", "========================================================================");
    printf("✅ CUDA-Q Quick Start Complete!\n");
    printf("%s\n", "========================================================================");
    printf("\nNext steps:\n");
    printf("1. Explore more examples in /root/Qallow/third_party/cuda-quantum/examples/\n");
    printf("2. Read the documentation: https://nvidia.github.io/cuda-quantum/\n");
    printf("3. Integrate CUDA-Q with Qallow phases\n");
    printf("4. Build hybrid quantum-classical algorithms\n");
    printf("\nFor more information, see: /root/Qallow/CUDA_Q_GUIDE.md\n");
    
    return 0;
}

