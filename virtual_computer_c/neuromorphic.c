#include "neuromorphic.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static void neuron_init(Neuron *n) {
    n->a = 0.02f;
    n->b = 0.20f;
    n->c = -65.0f;
    n->d = 8.0f;
    n->v = -65.0f;
    n->u = n->b * n->v;
    n->spiked = 0;
}

int neuro_init(NeuromorphicProcessor *np, size_t neurons, size_t synapses, float dt_ms) {
    if (!np) {
        return -1;
    }
    memset(np, 0, sizeof(*np));
    np->neurons = (Neuron *)calloc(neurons, sizeof(Neuron));
    np->synapses = (Synapse *)calloc(synapses, sizeof(Synapse));
    if (!np->neurons || !np->synapses) {
        neuro_free(np);
        return -2;
    }
    np->neuron_count = neurons;
    np->synapse_count = 0;
    np->dt_ms = dt_ms;
    for (size_t i = 0; i < neurons; ++i) {
        neuron_init(&np->neurons[i]);
    }
    return 0;
}

void neuro_free(NeuromorphicProcessor *np) {
    if (!np) {
        return;
    }
    free(np->neurons);
    free(np->synapses);
    memset(np, 0, sizeof(*np));
}

int neuro_connect(NeuromorphicProcessor *np, int pre, int post, float w, float delay_ms) {
    if (!np || !np->synapses || np->synapse_count >= np->neuron_count * np->neuron_count) {
        return -1;
    }
    if (pre < 0 || post < 0 || (size_t)pre >= np->neuron_count || (size_t)post >= np->neuron_count) {
        return -1;
    }
    Synapse *s = &np->synapses[np->synapse_count++];
    s->pre = pre;
    s->post = post;
    s->w = w;
    s->delay_ms = delay_ms;
    return 0;
}

int neuro_inject_spike(NeuromorphicProcessor *np, int neuron_id) {
    if (!np || (size_t)neuron_id >= np->neuron_count) {
        return -1;
    }
    np->neurons[neuron_id].spiked = 1;
    np->perf.spikes++;
    return 0;
}

int neuro_step(NeuromorphicProcessor *np, size_t steps) {
    if (!np || !np->neurons) {
        return -1;
    }
    for (size_t s = 0; s < steps; ++s) {
        for (size_t i = 0; i < np->neuron_count; ++i) {
            Neuron *n = &np->neurons[i];
            float I = n->spiked ? 1.0f : 0.0f;
            float dv = 0.04f * n->v * n->v + 5.0f * n->v + 140.0f - n->u + I;
            float du = n->a * (n->b * n->v - n->u);
            n->v += dv * (np->dt_ms * 0.5f);
            n->u += du * (np->dt_ms * 0.5f);
            n->v += dv * (np->dt_ms * 0.5f);
            n->u += du * (np->dt_ms * 0.5f);

            if (n->v >= 30.0f) {
                n->v = n->c;
                n->u += n->d;
                n->spiked = 1;
                np->perf.spikes++;
            } else {
                n->spiked = 0;
            }
        }
        for (size_t k = 0; k < np->synapse_count; ++k) {
            Synapse *sy = &np->synapses[k];
            if (np->neurons[sy->pre].spiked) {
                np->neurons[sy->post].v += sy->w;
            }
        }
        np->perf.steps++;
    }
    return 0;
}
