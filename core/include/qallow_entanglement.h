#ifndef QALLOW_ENTANGLEMENT_H
#define QALLOW_ENTANGLEMENT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    QALLOW_ENT_STATE_GHZ = 0,
    QALLOW_ENT_STATE_W = 1
} qallow_entanglement_state_t;

typedef struct {
    qallow_entanglement_state_t state;
    int qubits;
    int amplitude_count;
    double amplitudes[32];
    double fidelity;
    char backend[16];
} qallow_entanglement_snapshot_t;

int qallow_entanglement_generate(
    qallow_entanglement_snapshot_t* out,
    qallow_entanglement_state_t state,
    int qubits,
    int validate
);

const qallow_entanglement_snapshot_t* qallow_entanglement_get_cached(void);

const char* qallow_entanglement_state_name(qallow_entanglement_state_t state);
qallow_entanglement_state_t qallow_entanglement_state_from_string(const char* raw);

#ifdef __cplusplus
}
#endif

#endif
