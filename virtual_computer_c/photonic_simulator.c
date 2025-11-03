/*
 * Photonic Processor Simulator - C Implementation
 */

#include "photonic_simulator.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define INITIAL_PHOTONS_CAPACITY 4096
#define QUANTUM_EFFICIENCY 0.95

/**
 * Create and initialize photonic processor
 */
photonic_processor_t* pp_create(uint32_t num_waveguides, uint32_t num_gates) {
    photonic_processor_t *pp = (photonic_processor_t *)malloc(sizeof(photonic_processor_t));
    if (!pp) return NULL;
    
    pp->num_waveguides = num_waveguides;
    pp->num_gates = num_gates;
    
    /* Allocate waveguides */
    pp->waveguides = (optical_waveguide_t *)malloc(num_waveguides * sizeof(optical_waveguide_t));
    if (!pp->waveguides) {
        free(pp);
        return NULL;
    }
    pp->waveguides_capacity = num_waveguides;
    
    /* Initialize waveguides */
    srand(time(NULL));
    for (uint32_t i = 0; i < num_waveguides; i++) {
        pp->waveguides[i].waveguide_id = i;
        pp->waveguides[i].length_mm = 1.0 + (rand() / (double)RAND_MAX) * 9.0;
        pp->waveguides[i].propagation_loss_db_per_km = 0.2 +
            (rand() / (double)RAND_MAX - 0.5) * 0.1;
        pp->waveguides[i].chromatic_dispersion = 0.0;
    }
    
    /* Allocate gates */
    pp->gates = (photonic_gate_t *)malloc(num_gates * sizeof(photonic_gate_t));
    if (!pp->gates) {
        free(pp->waveguides);
        free(pp);
        return NULL;
    }
    pp->gates_capacity = num_gates;
    
    /* Initialize gates */
    for (uint32_t i = 0; i < num_gates; i++) {
        pp->gates[i].gate_id = i;
        pp->gates[i].gate_type = rand() % 5;
        
        /* Set input/output ports based on gate type */
        if (pp->gates[i].gate_type == GATE_BEAM_SPLITTER ||
            pp->gates[i].gate_type == GATE_MACH_ZEHNDER ||
            pp->gates[i].gate_type == GATE_OPTICAL_SWITCH) {
            pp->gates[i].input_ports = 2;
            pp->gates[i].output_ports = 2;
        } else {
            pp->gates[i].input_ports = 1;
            pp->gates[i].output_ports = 1;
        }
        
        pp->gates[i].insertion_loss_db = 0.3 + (rand() / (double)RAND_MAX) * 0.7;
        pp->gates[i].switching_time_ns = 10.0;
        pp->gates[i].modulation_efficiency = 0.95;
        pp->gates[i].is_active = true;
        pp->gates[i].switching_count = 0;
        pp->gates[i].last_switch_time = 0.0;
        pp->gates[i].output_power_dbm = 0.0;
    }
    
    /* Allocate photons */
    pp->photons = (photon_t *)malloc(INITIAL_PHOTONS_CAPACITY * sizeof(photon_t));
    if (!pp->photons) {
        free(pp->gates);
        free(pp->waveguides);
        free(pp);
        return NULL;
    }
    pp->num_photons = 0;
    pp->photons_capacity = INITIAL_PHOTONS_CAPACITY;
    
    /* Statistics */
    pp->photon_counter = 0;
    pp->total_photons_injected = 0;
    pp->total_photons_detected = 0;
    pp->total_switching_operations = 0;
    pp->total_propagation_distance_mm = 0.0;
    pp->current_time = 0.0;
    
    return pp;
}

/**
 * Destroy photonic processor
 */
void pp_destroy(photonic_processor_t *pp) {
    if (!pp) return;
    free(pp->waveguides);
    free(pp->gates);
    free(pp->photons);
    free(pp);
}

/**
 * Inject photons into the optical system
 */
uint32_t pp_inject_photons(photonic_processor_t *pp, uint32_t count,
                          double power_dbm, double wavelength_nm,
                          uint32_t *photon_ids, uint32_t max_ids) {
    if (!pp) return 0;
    
    uint32_t injected = 0;
    
    for (uint32_t i = 0; i < count && injected < max_ids; i++) {
        /* Resize if needed */
        if (pp->num_photons >= pp->photons_capacity) {
            pp->photons_capacity *= 2;
            pp->photons = (photon_t *)realloc(pp->photons,
                pp->photons_capacity * sizeof(photon_t));
            if (!pp->photons) return injected;
        }
        
        pp->photon_counter++;
        photon_t *photon = &pp->photons[pp->num_photons];
        
        photon->photon_id = pp->photon_counter;
        photon->wavelength_nm = wavelength_nm + (rand() / (double)RAND_MAX - 0.5) * 20.0;
        photon->power_dbm = power_dbm + (rand() / (double)RAND_MAX - 0.5) * 4.0;
        photon->phase = (rand() / (double)RAND_MAX) * 2.0 * M_PI;
        photon->position = 0;
        photon->timestamp = pp->current_time;
        
        photon_ids[injected] = pp->photon_counter;
        pp->num_photons++;
        pp->total_photons_injected++;
        injected++;
    }
    
    return injected;
}

/**
 * Propagate photon through waveguide
 */
bool pp_propagate_through_waveguide(photonic_processor_t *pp, uint32_t photon_id,
                                   uint32_t waveguide_id, double *propagation_time_ns) {
    if (!pp || !propagation_time_ns || waveguide_id >= pp->num_waveguides) {
        return false;
    }
    
    /* Find photon */
    photon_t *photon = NULL;
    for (uint32_t i = 0; i < pp->num_photons; i++) {
        if (pp->photons[i].photon_id == photon_id) {
            photon = &pp->photons[i];
            break;
        }
    }
    
    if (!photon) return false;
    
    optical_waveguide_t *wg = &pp->waveguides[waveguide_id];
    
    /* Calculate loss */
    double loss_db = (wg->length_mm / 1000000.0) * wg->propagation_loss_db_per_km;
    photon->power_dbm -= loss_db;
    
    /* Calculate propagation time */
    double speed_mm_per_ns = 200.0;  /* Speed of light in fiber */
    *propagation_time_ns = wg->length_mm / speed_mm_per_ns;
    
    /* Chromatic dispersion effect */
    double dispersion_penalty = wg->chromatic_dispersion * (photon->wavelength_nm - 1550.0);
    *propagation_time_ns += fabs(dispersion_penalty) / 1000.0;
    
    pp->total_propagation_distance_mm += wg->length_mm;
    
    return true;
}

/**
 * Apply photonic gate operation
 */
void pp_apply_gate_operation(photonic_processor_t *pp, const uint32_t *photon_ids,
                            uint32_t count, uint32_t gate_id) {
    if (!pp || !photon_ids || gate_id >= pp->num_gates) return;
    
    photonic_gate_t *gate = &pp->gates[gate_id];
    
    if (!gate->is_active) return;
    
    /* Find valid photons */
    uint32_t processed = 0;
    for (uint32_t i = 0; i < count && processed < gate->input_ports; i++) {
        uint32_t pid = photon_ids[i];
        
        /* Find photon */
        photon_t *photon = NULL;
        for (uint32_t j = 0; j < pp->num_photons; j++) {
            if (pp->photons[j].photon_id == pid) {
                photon = &pp->photons[j];
                break;
            }
        }
        
        if (!photon) continue;
        
        /* Apply gate operation */
        photon->power_dbm -= gate->insertion_loss_db;
        
        if (gate->gate_type == GATE_BEAM_SPLITTER) {
            photon->power_dbm -= 3.0;  /* 3dB split */
        } else if (gate->gate_type == GATE_MACH_ZEHNDER) {
            photon->phase += (rand() / (double)RAND_MAX) * M_PI;
        } else if (gate->gate_type == GATE_PHASE_MODULATOR) {
            double phase_shift = (rand() / (double)RAND_MAX) * 2.0 * M_PI;
            photon->phase = fmod(photon->phase + phase_shift, 2.0 * M_PI);
            photon->power_dbm -= gate->insertion_loss_db * gate->modulation_efficiency;
        } else if (gate->gate_type == GATE_OPTICAL_SWITCH) {
            photon->position = rand() % gate->output_ports;
        }
        
        processed++;
    }
    
    gate->switching_count++;
    pp->total_switching_operations++;
}

/**
 * Detect photons at output
 */
uint32_t pp_detect_photons(photonic_processor_t *pp, const uint32_t *photon_ids,
                          uint32_t count, uint32_t *detected_ids, uint32_t max_detected) {
    if (!pp || !photon_ids || !detected_ids) return 0;
    
    uint32_t detected = 0;
    double min_detectable_power = -40.0;  /* dBm */
    
    for (uint32_t i = 0; i < count && detected < max_detected; i++) {
        uint32_t pid = photon_ids[i];
        
        /* Find photon */
        photon_t *photon = NULL;
        for (uint32_t j = 0; j < pp->num_photons; j++) {
            if (pp->photons[j].photon_id == pid) {
                photon = &pp->photons[j];
                break;
            }
        }
        
        if (!photon) continue;
        
        /* Check if detectable */
        if (photon->power_dbm < min_detectable_power) continue;
        
        /* Random detection based on quantum efficiency */
        if ((rand() / (double)RAND_MAX) < QUANTUM_EFFICIENCY) {
            detected_ids[detected] = pid;
            detected++;
            pp->total_photons_detected++;
        }
    }
    
    return detected;
}

/**
 * Get processor statistics
 */
void pp_get_processor_stats(photonic_processor_t *pp, char *buffer, size_t buffer_size) {
    if (!pp || !buffer) return;
    
    double detection_efficiency = pp->total_photons_injected > 0 ?
        (double)pp->total_photons_detected / pp->total_photons_injected : 0.0;
    
    snprintf(buffer, buffer_size,
        "Photonic Processor Statistics\n"
        "  Waveguides: %u\n"
        "  Gates: %u\n"
        "  Photons Injected: %lu\n"
        "  Photons Detected: %lu\n"
        "  Detection Efficiency: %.2f%%\n"
        "  Switching Operations: %lu\n",
        pp->num_waveguides,
        pp->num_gates,
        pp->total_photons_injected,
        pp->total_photons_detected,
        detection_efficiency * 100.0,
        pp->total_switching_operations
    );
}

/**
 * Get gate statistics
 */
void pp_get_gate_stats(photonic_processor_t *pp, char *buffer, size_t buffer_size) {
    if (!pp || !buffer) return;
    
    uint64_t total_switching = 0;
    double avg_loss = 0.0;
    uint32_t active_gates = 0;
    
    for (uint32_t i = 0; i < pp->num_gates; i++) {
        total_switching += pp->gates[i].switching_count;
        avg_loss += pp->gates[i].insertion_loss_db;
        if (pp->gates[i].is_active) {
            active_gates++;
        }
    }
    
    avg_loss = pp->num_gates > 0 ? avg_loss / pp->num_gates : 0.0;
    
    snprintf(buffer, buffer_size,
        "Photonic Gate Statistics\n"
        "  Total Gates: %u\n"
        "  Active Gates: %u\n"
        "  Total Switching: %lu\n"
        "  Avg Insertion Loss: %.2f dB\n",
        pp->num_gates,
        active_gates,
        total_switching,
        avg_loss
    );
}

/**
 * Get waveguide statistics
 */
void pp_get_waveguide_stats(photonic_processor_t *pp, char *buffer, size_t buffer_size) {
    if (!pp || !buffer) return;
    
    double total_length = 0.0;
    double avg_loss = 0.0;
    
    for (uint32_t i = 0; i < pp->num_waveguides; i++) {
        total_length += pp->waveguides[i].length_mm;
        avg_loss += (pp->waveguides[i].length_mm / 1000000.0) *
                   pp->waveguides[i].propagation_loss_db_per_km;
    }
    
    avg_loss = pp->num_waveguides > 0 ? avg_loss / pp->num_waveguides : 0.0;
    
    snprintf(buffer, buffer_size,
        "Photonic Waveguide Statistics\n"
        "  Total Waveguides: %u\n"
        "  Total Length: %.1f mm\n"
        "  Avg Loss per Waveguide: %.2f dB\n"
        "  Total Propagation Distance: %.1f mm\n",
        pp->num_waveguides,
        total_length,
        avg_loss,
        pp->total_propagation_distance_mm
    );
}

/**
 * Print processor status
 */
void pp_print_status(photonic_processor_t *pp) {
    if (!pp) return;
    
    char stats_buf[512];
    char gates_buf[512];
    char waveguides_buf[512];
    
    pp_get_processor_stats(pp, stats_buf, sizeof(stats_buf));
    pp_get_gate_stats(pp, gates_buf, sizeof(gates_buf));
    pp_get_waveguide_stats(pp, waveguides_buf, sizeof(waveguides_buf));
    
    printf("\n");
    printf("================================================================================\n");
    printf("  Photonic Processor Status\n");
    printf("================================================================================\n");
    printf("%s\n", stats_buf);
    printf("%s\n", gates_buf);
    printf("%s\n", waveguides_buf);
    printf("================================================================================\n\n");
}
