#include "qallow_entanglement.h"
#include "qallow_metrics.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_usage(void) {
    printf("Usage: qallow run entangle [--state=ghz|w] [--qubits=N] [--validate]\n");
}

static void format_basis_label(int index, int qubits, char* out, size_t len) {
    if (!out || len == 0) {
        return;
    }
    for (int i = qubits - 1; i >= 0; --i) {
        if ((size_t)(qubits - 1 - i) >= len - 1) {
            break;
        }
        out[qubits - 1 - i] = (char)(((index >> i) & 0x1) ? '1' : '0');
    }
    size_t used = (size_t)qubits;
    if (used >= len) {
        used = len - 1;
    }
    out[used] = '\0';
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
    char label[16];
    for (int i = 0; i < snapshot.amplitude_count; ++i) {
        format_basis_label(i, snapshot.qubits, label, sizeof(label));
        printf("  |%s⟩ : %.10f\n", label, snapshot.amplitudes[i]);
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
