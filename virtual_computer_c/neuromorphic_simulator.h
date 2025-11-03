/*
 * Neuromorphic Processor Simulator - C Header
 * Spiking Neural Networks with STDP and event-based processing
 */

#ifndef NEUROMORPHIC_SIMULATOR_H
#define NEUROMORPHIC_SIMULATOR_H

#include <stdint.h>
#include <stdbool.h>
#include <time.h>

/* Neuron Type Enumeration */
typedef enum {
    NEURON_LIF = 0,           /* Leaky Integrate-and-Fire */
    NEURON_HODGKIN_HUXLEY = 1,
    NEURON_IZHIKEVICH = 2
} neuron_type_t;

/* Spike Event Structure */
typedef struct {
    uint32_t neuron_id;
    double timestamp;
    uint32_t source_layer;
} spike_event_t;

/* Neuron Structure */
typedef struct {
    uint32_t neuron_id;
    neuron_type_t neuron_type;
    uint32_t layer;
    
    /* LIF parameters */
    double membrane_potential;
    double threshold;
    double resting_potential;
    double tau_membrane;
    double tau_synaptic;
    
    /* Tracking */
    uint64_t spike_count;
    double last_spike_time;
    double synaptic_input;
    double refractory_period;
    bool in_refractory;
} neuron_t;

/* Synapse Structure with Plasticity */
typedef struct {
    uint32_t pre_neuron_id;
    uint32_t post_neuron_id;
    double weight;
    double delay;
    double learning_rate;
    double last_update;
    double pre_spike_trace;
    double post_spike_trace;
    bool plasticity_enabled;
} synapse_t;

/* Neuromorphic Processor Structure */
typedef struct {
    uint32_t num_neurons;
    uint32_t num_layers;
    
    /* Neurons */
    neuron_t *neurons;
    uint32_t neurons_capacity;
    
    /* Synapses */
    synapse_t *synapses;
    uint32_t num_synapses;
    uint32_t synapses_capacity;
    
    /* Spike tracking */
    spike_event_t *spike_log;
    uint32_t spike_log_size;
    uint32_t spike_log_capacity;
    
    double current_time;
    double time_step;
    
    /* Statistics */
    uint64_t total_spikes;
    uint64_t total_simulation_steps;
    double energy_consumed_uj;
    double latency_ms;
} neuromorphic_processor_t;

/* Function Declarations */

/**
 * Create and initialize neuromorphic processor
 */
neuromorphic_processor_t* nm_create(uint32_t num_neurons, uint32_t num_layers);

/**
 * Destroy neuromorphic processor
 */
void nm_destroy(neuromorphic_processor_t *nm);

/**
 * Create random connectivity between neurons
 */
void nm_create_connectivity(neuromorphic_processor_t *nm, double connectivity_ratio);

/**
 * Inject spikes into neurons (input)
 */
void nm_inject_spikes(neuromorphic_processor_t *nm, const uint32_t *neuron_ids,
                     uint32_t count, double current_time);

/**
 * Update neuron state using LIF model
 */
bool nm_update_neuron(neuromorphic_processor_t *nm, uint32_t neuron_id, 
                     double current_time);

/**
 * Propagate spikes through synapses
 */
void nm_propagate_spikes(neuromorphic_processor_t *nm, double current_time);

/**
 * Simulate one time step
 */
void nm_simulate_step(neuromorphic_processor_t *nm, double current_time, bool inject_input);

/**
 * Get average spike rate for a layer
 */
double nm_get_layer_spike_rate(neuromorphic_processor_t *nm, uint32_t layer);

/**
 * Get processor statistics
 */
void nm_get_stats(neuromorphic_processor_t *nm, char *buffer, size_t buffer_size);

/**
 * Get connectivity statistics
 */
void nm_get_connectivity_stats(neuromorphic_processor_t *nm, char *buffer, size_t buffer_size);

/**
 * Print processor status
 */
void nm_print_status(neuromorphic_processor_t *nm);

#endif /* NEUROMORPHIC_SIMULATOR_H */
