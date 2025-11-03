/*
 * photonic.h - Minimal photonic processor interface
 */

#ifndef PHOTONIC_H
#define PHOTONIC_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    PHOTONIC_GATE_BS,
    PHOTONIC_GATE_PHASE,
    PHOTONIC_GATE_MZI
} PhotonicGateType;

typedef struct {
    PhotonicGateType type;
    int in_a;
    int in_b;
    int out_a;
    int out_b;
    float theta;
    float eta;
} PhotonicGate;

typedef struct {
    float amplitude;
    float phase;
} PhotonState;

typedef struct {
    PhotonState *waveguides;
    size_t channels;
    PhotonicGate *gates;
    size_t gate_count;
    struct {
        uint64_t propagations;
    } perf;
} PhotonicProcessor;

int photonic_init(PhotonicProcessor *pp, size_t channels, size_t max_gates);
void photonic_free(PhotonicProcessor *pp);
int photonic_add_gate(PhotonicProcessor *pp, PhotonicGate gate);
int photonic_inject(PhotonicProcessor *pp, int channel, float amplitude, float phase);
int photonic_propagate(PhotonicProcessor *pp, size_t steps);

#endif /* PHOTONIC_H */
