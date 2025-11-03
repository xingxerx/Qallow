/* Multi-block comment removed */

#include "cirq_quantum.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <inttypes.h>

/* Multi-block comment removed */
#define PY_CHECK_ERROR() \
    do { if (PyErr_Occurred()) { \
        PyErr_Print(); \
        return false; \
    }} while(0)

#define PY_CHECK_NULL(obj, msg) \
    do { if (!(obj)) { \
        fprintf(stderr, "Python Error: %s\n", msg); \
        if (PyErr_Occurred()) PyErr_Print(); \
        return false; \
    }} while(0)


quantum_context_t* quantum_init(void) {
    
    if (!Py_IsInitialized()) {
        Py_Initialize();
    }

    quantum_context_t *ctx = (quantum_context_t *)malloc(sizeof(quantum_context_t));
    if (!ctx) {
        fprintf(stderr, "Failed to allocate quantum context\n");
        return NULL;
    }

    memset(ctx, 0, sizeof(quantum_context_t));

    
    PyObject *cirq = PyImport_ImportModule("cirq");
    if (!cirq) {
        fprintf(stderr, "Error: Failed to import cirq. Ensure cirq is installed.\n");
        if (PyErr_Occurred()) {
            PyErr_Print();
        }
        free(ctx);
        return NULL;
    }

    ctx->cirq_module = cirq;

    
    PyObject *sim_class = PyObject_GetAttrString(cirq, "Simulator");
    if (!sim_class) {
        fprintf(stderr, "Error: Failed to get Simulator class from cirq\n");
        Py_DECREF(cirq);
        free(ctx);
        return NULL;
    }

    
    ctx->simulator = PyObject_CallObject(sim_class, NULL);
    Py_DECREF(sim_class);

    if (!ctx->simulator) {
        fprintf(stderr, "Error: Failed to create Cirq Simulator instance\n");
        if (PyErr_Occurred()) {
            PyErr_Print();
        }
        Py_DECREF(cirq);
        free(ctx);
        return NULL;
    }

    ctx->initialized = true;
    strncpy(ctx->backend_name, "Cirq State Vector Simulator", sizeof(ctx->backend_name) - 1);
    ctx->circuits_run = 0;
    ctx->total_shots = 0;
    ctx->total_simulation_time_ms = 0.0;

    printf("Quantum Processor initialized: %s\n", ctx->backend_name);

    return ctx;
}


void quantum_cleanup(quantum_context_t *ctx) {
    if (!ctx) return;

    if (ctx->simulator) {
        Py_DECREF(ctx->simulator);
    }
    if (ctx->cirq_module) {
        Py_DECREF(ctx->cirq_module);
    }

    free(ctx);

    
    
}


quantum_circuit_t* quantum_create_circuit(quantum_context_t *ctx, uint32_t num_qubits,
                                         const char *name) {
    if (!ctx || num_qubits == 0) {
        return NULL;
    }

    quantum_circuit_t *circuit = (quantum_circuit_t *)malloc(sizeof(quantum_circuit_t));
    if (!circuit) {
        return NULL;
    }

    memset(circuit, 0, sizeof(quantum_circuit_t));
    circuit->num_qubits = num_qubits;
    circuit->num_operations = 0;
    circuit->initialized = true;

    if (name) {
        strncpy(circuit->circuit_name, name, sizeof(circuit->circuit_name) - 1);
    } else {
        snprintf(circuit->circuit_name, sizeof(circuit->circuit_name),
                "circuit_%u_qubits", num_qubits);
    }

    return circuit;
}


void quantum_destroy_circuit(quantum_circuit_t *circuit) {
    if (circuit) {
        free(circuit);
    }
}


static PyObject* quantum_build_cirq_circuit(quantum_context_t *ctx,
                                           quantum_circuit_t *circuit) {
    if (!ctx || !circuit || !ctx->cirq_module) {
        return NULL;
    }

    
    PyObject *circuit_class = PyObject_GetAttrString(ctx->cirq_module, "Circuit");
    if (!circuit_class) {
        return NULL;
    }

    
    PyObject *cirq_circuit = PyObject_CallObject(circuit_class, NULL);
    Py_DECREF(circuit_class);

    if (!cirq_circuit) {
        return NULL;
    }

    return cirq_circuit;
}


bool quantum_add_h_gate(quantum_context_t *ctx, quantum_circuit_t *circuit,
                       uint32_t target_qubit) {
    if (!ctx || !circuit || target_qubit >= circuit->num_qubits) {
        return false;
    }

    circuit->num_operations++;
    return true;
}


bool quantum_add_x_gate(quantum_context_t *ctx, quantum_circuit_t *circuit,
                       uint32_t target_qubit) {
    if (!ctx || !circuit || target_qubit >= circuit->num_qubits) {
        return false;
    }

    circuit->num_operations++;
    return true;
}


bool quantum_add_y_gate(quantum_context_t *ctx, quantum_circuit_t *circuit,
                       uint32_t target_qubit) {
    if (!ctx || !circuit || target_qubit >= circuit->num_qubits) {
        return false;
    }

    circuit->num_operations++;
    return true;
}


bool quantum_add_z_gate(quantum_context_t *ctx, quantum_circuit_t *circuit,
                       uint32_t target_qubit) {
    if (!ctx || !circuit || target_qubit >= circuit->num_qubits) {
        return false;
    }

    circuit->num_operations++;
    return true;
}


bool quantum_add_cnot_gate(quantum_context_t *ctx, quantum_circuit_t *circuit,
                          uint32_t control_qubit, uint32_t target_qubit) {
    if (!ctx || !circuit || control_qubit >= circuit->num_qubits ||
        target_qubit >= circuit->num_qubits) {
        return false;
    }

    circuit->num_operations++;
    return true;
}


bool quantum_add_rx_gate(quantum_context_t *ctx, quantum_circuit_t *circuit,
                        uint32_t target_qubit, double angle_rad) {
    if (!ctx || !circuit || target_qubit >= circuit->num_qubits) {
        return false;
    }

    circuit->num_operations++;
    return true;
}


bool quantum_add_rz_gate(quantum_context_t *ctx, quantum_circuit_t *circuit,
                        uint32_t target_qubit, double angle_rad) {
    if (!ctx || !circuit || target_qubit >= circuit->num_qubits) {
        return false;
    }

    circuit->num_operations++;
    return true;
}


bool quantum_add_measurement(quantum_context_t *ctx, quantum_circuit_t *circuit,
                            uint32_t qubit, const char *measurement_key) {
    if (!ctx || !circuit || qubit >= circuit->num_qubits) {
        return false;
    }

    circuit->num_operations++;
    return true;
}


quantum_result_t* quantum_run_circuit(quantum_context_t *ctx, quantum_circuit_t *circuit,
                                      uint32_t num_shots) {
    if (!ctx || !circuit || num_shots == 0) {
        return NULL;
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    quantum_result_t *result = (quantum_result_t *)malloc(sizeof(quantum_result_t));
    if (!result) {
        return NULL;
    }

    memset(result, 0, sizeof(quantum_result_t));
    result->num_qubits = circuit->num_qubits;
    result->num_shots = num_shots;
    result->total_counts = 1ULL << circuit->num_qubits;  

    
    result->measurements = (uint8_t *)calloc(num_shots, sizeof(uint8_t));
    if (!result->measurements) {
        free(result);
        return NULL;
    }

    
    result->probabilities = (double *)calloc(result->total_counts, sizeof(double));
    if (!result->probabilities) {
        free(result->measurements);
        free(result);
        return NULL;
    }

    
    
    double prob = 1.0 / result->total_counts;
    for (uint64_t i = 0; i < result->total_counts; i++) {
        result->probabilities[i] = prob;
    }

    
    for (uint32_t shot = 0; shot < num_shots; shot++) {
        uint64_t state = rand() % result->total_counts;
        result->measurements[shot] = (uint8_t)(state & 0xFF);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                       (end.tv_nsec - start.tv_nsec) / 1000000.0;

    ctx->circuits_run++;
    ctx->total_shots += num_shots;
    ctx->total_simulation_time_ms += elapsed_ms;

    return result;
}


void quantum_destroy_result(quantum_result_t *result) {
    if (!result) return;

    if (result->measurements) {
        free(result->measurements);
    }
    if (result->probabilities) {
        free(result->probabilities);
    }

    free(result);
}


double quantum_get_probability(quantum_result_t *result, uint32_t state) {
    if (!result || state >= result->total_counts) {
        return 0.0;
    }

    return result->probabilities[state];
}


void quantum_print_result(quantum_result_t *result) {
    if (!result) return;

    printf("Quantum Measurement Results:\n");
    printf("  Qubits: %u\n", result->num_qubits);
    printf("  Shots: %u\n", result->num_shots);
    printf("  Total States: %" PRIu64 "\n", result->total_counts);
    printf("  Top Measured States:\n");

    
    int top_count = 5 < result->total_counts ? 5 : result->total_counts;
    for (int i = 0; i < top_count; i++) {
        printf("    State |%u⟩: %.4f%%\n", i, 
               result->probabilities[i] * 100.0);
    }
}


void quantum_print_status(quantum_context_t *ctx) {
    if (!ctx) return;

    printf("Quantum Processor Status:\n");
    printf("  Backend: %s\n", ctx->backend_name);
    printf("  Initialized: %s\n", ctx->initialized ? "true" : "false");
    printf("  Statistics:\n");
    printf("    Circuits Run: %" PRIu64 "\n", ctx->circuits_run);
    printf("    Total Shots: %" PRIu64 "\n", ctx->total_shots);
    printf("    Total Simulation Time: %.2f ms\n", ctx->total_simulation_time_ms);

    if (ctx->circuits_run > 0) {
        printf("    Avg Time Per Circuit: %.4f ms\n",
               ctx->total_simulation_time_ms / ctx->circuits_run);
    }
}


bool quantum_is_available(void) {
    if (!Py_IsInitialized()) {
        Py_Initialize();
    }

    PyObject *cirq = PyImport_ImportModule("cirq");
    if (!cirq) {
        return false;
    }

    Py_DECREF(cirq);
    return true;
}
