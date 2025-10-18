/**
 * @file sim_adapter.c
 * @brief Simulation adapter for internal testing
 * 
 * Generates synthetic data streams for testing without external dependencies
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "ingest.h"

// Simulation state
typedef struct {
    double coherence;
    double decoherence;
    double stability;
    int tick;
    double noise_level;
} sim_state_t;

static sim_state_t sim_state = {
    .coherence = 0.9984,
    .decoherence = 0.000010,
    .stability = 0.9984,
    .tick = 0,
    .noise_level = 0.001
};

/**
 * Generate synthetic coherence value
 */
static double sim_generate_coherence(void) {
    // Oscillate around 0.9984 with small noise
    double noise = (rand() % 1000) / 1000000.0 - 0.0005;
    double value = 0.9984 + noise;
    return (value < 0.0) ? 0.0 : (value > 1.0) ? 1.0 : value;
}

/**
 * Generate synthetic decoherence value
 */
static double sim_generate_decoherence(void) {
    // Very small value with noise
    double noise = (rand() % 100) / 10000000.0;
    double value = 0.000010 + noise;
    return (value < 0.0) ? 0.0 : value;
}

/**
 * Generate synthetic stability value
 */
static double sim_generate_stability(void) {
    // Oscillate around 0.9984
    double noise = (rand() % 1000) / 1000000.0 - 0.0005;
    double value = 0.9984 + noise;
    return (value < 0.0) ? 0.0 : (value > 1.0) ? 1.0 : value;
}

/**
 * Generate synthetic feedback score
 */
static double sim_generate_feedback(void) {
    // Human feedback score (0.0-1.0)
    return 0.75 + (rand() % 200) / 1000.0 - 0.1;
}

/**
 * Generate a synthetic data packet
 */
int sim_adapter_generate_packet(ingest_packet_type_t type, 
                                ingest_packet_t* packet) {
    if (!packet) return -1;
    
    memset(packet, 0, sizeof(ingest_packet_t));
    
    packet->timestamp = time(NULL);
    packet->type = type;
    packet->confidence = 0.99;
    strncpy(packet->source, "sim_adapter", sizeof(packet->source) - 1);
    
    switch (type) {
        case INGEST_TYPE_TELEMETRY:
            packet->value = sim_generate_coherence();
            strncpy(packet->metadata, "coherence_measurement", sizeof(packet->metadata) - 1);
            break;
            
        case INGEST_TYPE_SENSOR:
            packet->value = sim_generate_decoherence();
            strncpy(packet->metadata, "decoherence_measurement", sizeof(packet->metadata) - 1);
            break;
            
        case INGEST_TYPE_FEEDBACK:
            packet->value = sim_generate_feedback();
            strncpy(packet->metadata, "hitl_feedback", sizeof(packet->metadata) - 1);
            break;
            
        case INGEST_TYPE_CONTROL:
            packet->value = 1.0; // Control enabled
            strncpy(packet->metadata, "control_active", sizeof(packet->metadata) - 1);
            break;
            
        default:
            packet->value = sim_generate_stability();
            strncpy(packet->metadata, "generic_measurement", sizeof(packet->metadata) - 1);
            break;
    }
    
    return 0;
}

/**
 * Poll simulation and push packet to ingestion manager
 */
int sim_adapter_poll(ingest_manager_t* mgr, ingest_packet_type_t type) {
    if (!mgr) return -1;
    
    ingest_packet_t packet;
    
    // Generate synthetic packet
    if (sim_adapter_generate_packet(type, &packet) != 0) {
        return -1;
    }
    
    // Push to ingestion manager
    if (ingest_push_packet(mgr, &packet) != 0) {
        return -1;
    }
    
    printf("[SIM_ADAPTER] Packet generated: type=%d, value=%.6f\n", 
           type, packet.value);
    
    return 0;
}

/**
 * Run continuous simulation
 */
int sim_adapter_run_cycle(ingest_manager_t* mgr) {
    if (!mgr) return -1;
    
    // Generate packets for each type
    sim_adapter_poll(mgr, INGEST_TYPE_TELEMETRY);
    sim_adapter_poll(mgr, INGEST_TYPE_SENSOR);
    sim_adapter_poll(mgr, INGEST_TYPE_FEEDBACK);
    
    sim_state.tick++;
    return 0;
}

/**
 * Initialize simulation adapter
 */
void sim_adapter_init(void) {
    srand((unsigned int)time(NULL));
    printf("[SIM_ADAPTER] Simulation adapter initialized\n");
}

/**
 * Cleanup simulation adapter
 */
void sim_adapter_cleanup(void) {
    printf("[SIM_ADAPTER] Simulation adapter cleaned up\n");
}

/**
 * Get simulation state
 */
void sim_adapter_get_state(sim_state_t* state) {
    if (state) {
        *state = sim_state;
    }
}

