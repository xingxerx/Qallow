#include "qallow_entanglement.h"
#include "qallow_metrics.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_usage(void) {
    printf("Usage: qallow run entangle [--state=ghz|w] [--qubits=N] [--validate]\n");
}

int qallow_cmd_entangle(int argc, char** argv) {
    const char* state_arg = "ghz";
    int qubits = 4;
    int validate = 0;

    for (int i = 0; i < argc; ++i) {
        const char* arg = argv[i];
        if (!arg) {
            continue;
        }
        if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            print_usage();
            return 0;
        }
        if (strncmp(arg, "--state=", 8) == 0) {
            state_arg = arg + 8;
            continue;
        }
        if (strncmp(arg, "--qubits=", 9) == 0) {
            qubits = atoi(arg + 9);
            continue;
        }
        if (strcmp(arg, "--validate") == 0) {
            validate = 1;
            continue;
        }
        fprintf(stderr, "[ENTANGLE] Unknown option: %s\n", arg);
        print_usage();
        return 1;
    }

    if (qubits < 2 || qubits > 5) {
        fprintf(stderr, "[ENTANGLE] qubits must be between 2 and 5 (received %d)\n", qubits);
        return 1;
    }

    qallow_entanglement_state_t state = qallow_entanglement_state_from_string(state_arg);
    qallow_entanglement_snapshot_t snapshot;
    if (qallow_entanglement_generate(&snapshot, state, qubits, validate) != 0) {
        fprintf(stderr, "[ENTANGLE] Failed to generate entanglement snapshot. Ensure QuTiP is installed.\n");
        return 2;
    }

    printf("\n╔════════════════════════════════════════════════════════════╗\n");
    printf("║          QALLOW QUANTUM ENTANGLEMENT SNAPSHOT             ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    printf("State:            %s\n", qallow_entanglement_state_name(snapshot.state));
    printf("Qubits:           %d\n", snapshot.qubits);
    printf("Validation:       %s\n", validate ? "enabled" : "disabled");
    printf("Backend:          %s\n", snapshot.backend);
    printf("Fidelity:         %.6f\n", snapshot.fidelity);
    printf("Amplitudes:       %d\n", snapshot.amplitude_count);

    printf("\nProbabilities:\n");
    for (int i = 0; i < snapshot.amplitude_count; ++i) {
        printf("  |%0*d⟩ : %.10f\n", snapshot.qubits, i, snapshot.amplitudes[i]);
    }

    const qallow_run_metrics_t* metrics = qallow_get_last_run_metrics();
    if (metrics && metrics->tick_count > 0) {
        printf("\nRecent VM Run Metrics:\n");
        printf("  ticks:            %d\n", metrics->tick_count);
        printf("  cuda_enabled:     %s\n", metrics->cuda_enabled ? "yes" : "no");
        if (metrics->reached_equilibrium) {
            printf("  equilibrium_tick: %d\n", metrics->equilibrium_tick);
        }
        printf("  final_coherence:  %.6f\n", metrics->final_coherence);
        printf("  final_decoherence %.6f\n", metrics->final_decoherence);
    }

    printf("\nHint: export QALLOW_ENTANGLEMENT_BOOTSTRAP=ghz to seed pocket simulations.\n\n");
    return 0;
}
