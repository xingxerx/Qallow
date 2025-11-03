#include "photonic.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

int photonic_init(PhotonicProcessor *pp, size_t channels, size_t max_gates) {
    if (!pp) {
        return -1;
    }
    memset(pp, 0, sizeof(*pp));
    pp->waveguides = (PhotonState *)calloc(channels, sizeof(PhotonState));
    pp->gates = (PhotonicGate *)calloc(max_gates, sizeof(PhotonicGate));
    if (!pp->waveguides || !pp->gates) {
        photonic_free(pp);
        return -2;
    }
    pp->channels = channels;
    pp->gate_count = 0;
    return 0;
}

void photonic_free(PhotonicProcessor *pp) {
    if (!pp) {
        return;
    }
    free(pp->waveguides);
    free(pp->gates);
    memset(pp, 0, sizeof(*pp));
}

int photonic_add_gate(PhotonicProcessor *pp, PhotonicGate gate) {
    if (!pp || !pp->gates) {
        return -1;
    }
    pp->gates[pp->gate_count++] = gate;
    return 0;
}

int photonic_inject(PhotonicProcessor *pp, int channel, float amplitude, float phase) {
    if (!pp || !pp->waveguides || channel < 0 || (size_t)channel >= pp->channels) {
        return -1;
    }
    pp->waveguides[channel].amplitude = amplitude;
    pp->waveguides[channel].phase = phase;
    return 0;
}

static void apply_gate(PhotonicProcessor *pp, PhotonicGate *g) {
    PhotonState *a = &pp->waveguides[g->in_a];
    PhotonState *b = &pp->waveguides[g->in_b];
    PhotonState *oa = &pp->waveguides[g->out_a];
    PhotonState *ob = &pp->waveguides[g->out_b];

    switch (g->type) {
        case PHOTONIC_GATE_BS: {
            float t = g->eta;
            float r = 1.0f - t;
            oa->amplitude = t * a->amplitude + r * b->amplitude;
            ob->amplitude = r * a->amplitude + t * b->amplitude;
            oa->phase = a->phase;
            ob->phase = b->phase;
            break;
        }
        case PHOTONIC_GATE_PHASE: {
            oa->amplitude = a->amplitude;
            oa->phase = a->phase + g->theta;
            ob->amplitude = b->amplitude;
            ob->phase = b->phase;
            break;
        }
        case PHOTONIC_GATE_MZI: {
            float inter = a->amplitude + b->amplitude * cosf(g->theta);
            oa->amplitude = inter;
            ob->amplitude = fabsf(a->amplitude - b->amplitude);
            oa->phase = a->phase + g->theta;
            ob->phase = b->phase - g->theta;
            break;
        }
        default:
            break;
    }
}

int photonic_propagate(PhotonicProcessor *pp, size_t steps) {
    if (!pp) {
        return -1;
    }
    for (size_t s = 0; s < steps; ++s) {
        for (size_t i = 0; i < pp->gate_count; ++i) {
            apply_gate(pp, &pp->gates[i]);
        }
        pp->perf.propagations++;
    }
    return 0;
}
