/*
 * neuromorphic.h - Minimal neuromorphic processor interface
 */

#ifndef NEUROMORPHIC_H
#define NEUROMORPHIC_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    float v;
    float u;
    float a;
    float b;
    float c;
    float d;
    uint8_t spiked;
} Neuron;

typedef struct {
    int pre;
    int post;
    float w;
    float delay_ms;
} Synapse;

typedef struct {
    Neuron *neurons;
    size_t neuron_count;
    Synapse *synapses;
    size_t synapse_count;
    float dt_ms;
    struct {
        uint64_t steps;
        uint64_t spikes;
    } perf;
} NeuromorphicProcessor;

int neuro_init(NeuromorphicProcessor *np, size_t neurons, size_t synapses, float dt_ms);
void neuro_free(NeuromorphicProcessor *np);
int neuro_connect(NeuromorphicProcessor *np, int pre, int post, float w, float delay_ms);
int neuro_inject_spike(NeuromorphicProcessor *np, int neuron_id);
int neuro_step(NeuromorphicProcessor *np, size_t steps);

#endif /* NEUROMORPHIC_H */
