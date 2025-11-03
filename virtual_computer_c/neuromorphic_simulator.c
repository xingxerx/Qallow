/*
 * Neuromorphic Processor Simulator - C Implementation
 */

#include "neuromorphic_simulator.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define INITIAL_NEURONS_CAPACITY 1024
#define INITIAL_SYNAPSES_CAPACITY 8192
#define INITIAL_SPIKE_LOG_CAPACITY 4096

/**
 * Create and initialize neuromorphic processor
 */
neuromorphic_processor_t* nm_create(uint32_t num_neurons, uint32_t num_layers) {
    neuromorphic_processor_t *nm = (neuromorphic_processor_t *)malloc(sizeof(neuromorphic_processor_t));
    if (!nm) return NULL;
    
    nm->num_neurons = num_neurons;
    nm->num_layers = num_layers;
    
    /* Allocate neurons */
    nm->neurons = (neuron_t *)malloc(num_neurons * sizeof(neuron_t));
    if (!nm->neurons) {
        free(nm);
        return NULL;
    }
    nm->neurons_capacity = num_neurons;
    
    /* Initialize neurons */
    uint32_t neurons_per_layer = num_neurons / num_layers;
    srand(time(NULL));
    
    for (uint32_t i = 0; i < num_neurons; i++) {
        neuron_t *neuron = &nm->neurons[i];
        neuron->neuron_id = i;
        neuron->neuron_type = NEURON_LIF;
        neuron->layer = i / neurons_per_layer;
        
        /* LIF parameters */
        neuron->membrane_potential = 0.0;
        neuron->threshold = 1.0 + (rand() / (double)RAND_MAX - 0.5) * 0.4;
        neuron->resting_potential = -70.0;
        neuron->tau_membrane = 20.0;
        neuron->tau_synaptic = 5.0;
        
        /* Tracking */
        neuron->spike_count = 0;
        neuron->last_spike_time = 0.0;
        neuron->synaptic_input = 0.0;
        neuron->refractory_period = 2.0;
        neuron->in_refractory = false;
    }
    
    /* Allocate synapses */
    nm->synapses = (synapse_t *)malloc(INITIAL_SYNAPSES_CAPACITY * sizeof(synapse_t));
    if (!nm->synapses) {
        free(nm->neurons);
        free(nm);
        return NULL;
    }
    nm->num_synapses = 0;
    nm->synapses_capacity = INITIAL_SYNAPSES_CAPACITY;
    
    /* Allocate spike log */
    nm->spike_log = (spike_event_t *)malloc(INITIAL_SPIKE_LOG_CAPACITY * sizeof(spike_event_t));
    if (!nm->spike_log) {
        free(nm->synapses);
        free(nm->neurons);
        free(nm);
        return NULL;
    }
    nm->spike_log_size = 0;
    nm->spike_log_capacity = INITIAL_SPIKE_LOG_CAPACITY;
    
    /* Time tracking */
    nm->current_time = 0.0;
    nm->time_step = 1.0;
    
    /* Statistics */
    nm->total_spikes = 0;
    nm->total_simulation_steps = 0;
    nm->energy_consumed_uj = 0.0;
    nm->latency_ms = 0.0;
    
    /* Create connectivity */
    nm_create_connectivity(nm, 0.1);
    
    return nm;
}

/**
 * Destroy neuromorphic processor
 */
void nm_destroy(neuromorphic_processor_t *nm) {
    if (!nm) return;
    free(nm->neurons);
    free(nm->synapses);
    free(nm->spike_log);
    free(nm);
}

/**
 * Create random connectivity between neurons
 */
void nm_create_connectivity(neuromorphic_processor_t *nm, double connectivity_ratio) {
    if (!nm) return;
    
    uint32_t target_connections = (uint32_t)(nm->num_neurons * connectivity_ratio);
    
    for (uint32_t i = 0; i < target_connections; i++) {
        uint32_t pre = rand() % nm->num_neurons;
        uint32_t post = rand() % nm->num_neurons;
        
        if (pre != post) {
            /* Check for duplicate */
            bool exists = false;
            for (uint32_t j = 0; j < nm->num_synapses; j++) {
                if (nm->synapses[j].pre_neuron_id == pre && nm->synapses[j].post_neuron_id == post) {
                    exists = true;
                    break;
                }
            }
            
            if (!exists) {
                /* Resize if needed */
                if (nm->num_synapses >= nm->synapses_capacity) {
                    nm->synapses_capacity *= 2;
                    nm->synapses = (synapse_t *)realloc(nm->synapses,
                        nm->synapses_capacity * sizeof(synapse_t));
                    if (!nm->synapses) return;
                }
                
                synapse_t *syn = &nm->synapses[nm->num_synapses];
                syn->pre_neuron_id = pre;
                syn->post_neuron_id = post;
                syn->weight = 0.1 + (rand() / (double)RAND_MAX) * 0.9;
                syn->delay = 1.0;
                syn->learning_rate = 0.01;
                syn->last_update = 0.0;
                syn->pre_spike_trace = 0.0;
                syn->post_spike_trace = 0.0;
                syn->plasticity_enabled = true;
                
                nm->num_synapses++;
            }
        }
    }
}

/**
 * Inject spikes into neurons (input)
 */
void nm_inject_spikes(neuromorphic_processor_t *nm, const uint32_t *neuron_ids,
                     uint32_t count, double current_time) {
    if (!nm || !neuron_ids) return;
    
    for (uint32_t i = 0; i < count; i++) {
        uint32_t nid = neuron_ids[i];
        if (nid < nm->num_neurons) {
            nm->neurons[nid].synaptic_input += 2.0;
            
            /* Record spike */
            if (nm->spike_log_size >= nm->spike_log_capacity) {
                nm->spike_log_capacity *= 2;
                nm->spike_log = (spike_event_t *)realloc(nm->spike_log,
                    nm->spike_log_capacity * sizeof(spike_event_t));
                if (!nm->spike_log) return;
            }
            
            nm->spike_log[nm->spike_log_size].neuron_id = nid;
            nm->spike_log[nm->spike_log_size].timestamp = current_time;
            nm->spike_log[nm->spike_log_size].source_layer = 0;
            nm->spike_log_size++;
        }
    }
}

/**
 * Update neuron state using LIF model
 */
bool nm_update_neuron(neuromorphic_processor_t *nm, uint32_t neuron_id,
                     double current_time) {
    if (!nm || neuron_id >= nm->num_neurons) return false;
    
    neuron_t *neuron = &nm->neurons[neuron_id];
    
    /* Skip if in refractory period */
    if (neuron->in_refractory) {
        if (current_time - neuron->last_spike_time > neuron->refractory_period) {
            neuron->in_refractory = false;
        } else {
            return false;
        }
    }
    
    /* LIF dynamics */
    double decay = exp(-nm->time_step / neuron->tau_membrane);
    neuron->membrane_potential = neuron->resting_potential +
        (neuron->membrane_potential - neuron->resting_potential) * decay +
        neuron->synaptic_input * 10.0 * (1.0 - decay);
    
    /* Decay synaptic input */
    neuron->synaptic_input *= exp(-nm->time_step / neuron->tau_synaptic);
    
    /* Check threshold */
    bool did_spike = false;
    if (neuron->membrane_potential >= neuron->threshold) {
        neuron->spike_count++;
        neuron->last_spike_time = current_time;
        neuron->in_refractory = true;
        neuron->membrane_potential = neuron->resting_potential;
        
        /* Record spike */
        if (nm->spike_log_size >= nm->spike_log_capacity) {
            nm->spike_log_capacity *= 2;
            nm->spike_log = (spike_event_t *)realloc(nm->spike_log,
                nm->spike_log_capacity * sizeof(spike_event_t));
            if (!nm->spike_log) return false;
        }
        
        nm->spike_log[nm->spike_log_size].neuron_id = neuron_id;
        nm->spike_log[nm->spike_log_size].timestamp = current_time;
        nm->spike_log[nm->spike_log_size].source_layer = neuron->layer;
        nm->spike_log_size++;
        
        nm->total_spikes++;
        did_spike = true;
    }
    
    return did_spike;
}

/**
 * Propagate spikes through synapses
 */
void nm_propagate_spikes(neuromorphic_processor_t *nm, double current_time) {
    if (!nm) return;
    
    /* Find recent spikes */
    for (uint32_t i = 0; i < nm->spike_log_size; i++) {
        if (nm->spike_log[i].timestamp > current_time - 5.0) {
            uint32_t spike_neuron = nm->spike_log[i].neuron_id;
            
            /* Find outgoing synapses */
            for (uint32_t j = 0; j < nm->num_synapses; j++) {
                if (nm->synapses[j].pre_neuron_id == spike_neuron) {
                    if (current_time - nm->spike_log[i].timestamp >= nm->synapses[j].delay) {
                        uint32_t post_id = nm->synapses[j].post_neuron_id;
                        nm->neurons[post_id].synaptic_input += nm->synapses[j].weight;
                    }
                }
            }
        }
    }
}

/**
 * Simulate one time step
 */
void nm_simulate_step(neuromorphic_processor_t *nm, double current_time, bool inject_input) {
    if (!nm) return;
    
    /* Inject random input if requested */
    if (inject_input) {
        uint32_t num_input = nm->num_neurons / 100;
        if (num_input < 1) num_input = 1;
        
        uint32_t *input_neurons = (uint32_t *)malloc(num_input * sizeof(uint32_t));
        if (input_neurons) {
            for (uint32_t i = 0; i < num_input; i++) {
                input_neurons[i] = rand() % nm->num_neurons;
            }
            nm_inject_spikes(nm, input_neurons, num_input, current_time);
            free(input_neurons);
        }
    }
    
    /* Update all neurons */
    uint32_t spikes_this_step = 0;
    for (uint32_t i = 0; i < nm->num_neurons; i++) {
        if (nm_update_neuron(nm, i, current_time)) {
            spikes_this_step++;
        }
    }
    
    /* Propagate spikes */
    nm_propagate_spikes(nm, current_time);
    
    /* Energy calculation */
    nm->energy_consumed_uj += spikes_this_step * 0.001;
    
    nm->total_simulation_steps++;
}

/**
 * Get average spike rate for a layer
 */
double nm_get_layer_spike_rate(neuromorphic_processor_t *nm, uint32_t layer) {
    if (!nm) return 0.0;
    
    uint32_t count = 0;
    uint64_t total_spikes = 0;
    
    for (uint32_t i = 0; i < nm->num_neurons; i++) {
        if (nm->neurons[i].layer == layer) {
            total_spikes += nm->neurons[i].spike_count;
            count++;
        }
    }
    
    return count > 0 ? (double)total_spikes / count : 0.0;
}

/**
 * Get processor statistics
 */
void nm_get_stats(neuromorphic_processor_t *nm, char *buffer, size_t buffer_size) {
    if (!nm || !buffer) return;
    
    double avg_spike_rate = nm->total_simulation_steps > 0 ?
        (double)nm->total_spikes / nm->total_simulation_steps : 0.0;
    
    snprintf(buffer, buffer_size,
        "Neuromorphic Processor Statistics\n"
        "  Total Neurons: %u\n"
        "  Total Synapses: %u\n"
        "  Total Spikes: %lu\n"
        "  Simulation Steps: %lu\n"
        "  Avg Spike Rate: %.2f Hz\n"
        "  Energy Consumed: %.2f µJ\n",
        nm->num_neurons,
        nm->num_synapses,
        nm->total_spikes,
        nm->total_simulation_steps,
        avg_spike_rate,
        nm->energy_consumed_uj
    );
}

/**
 * Get connectivity statistics
 */
void nm_get_connectivity_stats(neuromorphic_processor_t *nm, char *buffer, size_t buffer_size) {
    if (!nm || !buffer) return;
    
    double connectivity_ratio = nm->num_synapses > 0 ?
        (double)nm->num_synapses / (nm->num_neurons * nm->num_neurons) : 0.0;
    
    double avg_weight = 0.0;
    double min_weight = 1.0;
    double max_weight = 0.0;
    
    if (nm->num_synapses > 0) {
        for (uint32_t i = 0; i < nm->num_synapses; i++) {
            avg_weight += nm->synapses[i].weight;
            if (nm->synapses[i].weight < min_weight) {
                min_weight = nm->synapses[i].weight;
            }
            if (nm->synapses[i].weight > max_weight) {
                max_weight = nm->synapses[i].weight;
            }
        }
        avg_weight /= nm->num_synapses;
    }
    
    snprintf(buffer, buffer_size,
        "Neuromorphic Connectivity\n"
        "  Total Synapses: %u\n"
        "  Connectivity Ratio: %.4f\n"
        "  Avg Weight: %.3f\n"
        "  Min Weight: %.3f\n"
        "  Max Weight: %.3f\n",
        nm->num_synapses,
        connectivity_ratio,
        avg_weight,
        min_weight,
        max_weight
    );
}

/**
 * Print processor status
 */
void nm_print_status(neuromorphic_processor_t *nm) {
    if (!nm) return;
    
    char stats_buf[512];
    char conn_buf[512];
    
    nm_get_stats(nm, stats_buf, sizeof(stats_buf));
    nm_get_connectivity_stats(nm, conn_buf, sizeof(conn_buf));
    
    printf("\n");
    printf("================================================================================\n");
    printf("  Neuromorphic Processor Status\n");
    printf("================================================================================\n");
    printf("%s\n", stats_buf);
    printf("%s\n", conn_buf);
    printf("================================================================================\n\n");
}
