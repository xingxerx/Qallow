/*
 * Photonic Processor Simulator - C Header
 * Integrated optical computing with photonic gates and waveguides
 */

#ifndef PHOTONIC_SIMULATOR_H
#define PHOTONIC_SIMULATOR_H

#include <stdint.h>
#include <stdbool.h>

/* Photonic Gate Types */
typedef enum {
    GATE_MACH_ZEHNDER = 0,
    GATE_BEAM_SPLITTER = 1,
    GATE_PHASE_MODULATOR = 2,
    GATE_OPTICAL_SWITCH = 3,
    GATE_DIRECTIONAL_COUPLER = 4
} photonic_gate_type_t;

/* Photon Structure */
typedef struct {
    uint32_t photon_id;
    double wavelength_nm;
    double power_dbm;
    double phase;
    uint32_t position;
    double timestamp;
} photon_t;

/* Optical Waveguide Structure */
typedef struct {
    uint32_t waveguide_id;
    double length_mm;
    double propagation_loss_db_per_km;
    double chromatic_dispersion;
} optical_waveguide_t;

/* Photonic Gate Structure */
typedef struct {
    uint32_t gate_id;
    photonic_gate_type_t gate_type;
    uint32_t input_ports;
    uint32_t output_ports;
    
    double insertion_loss_db;
    double switching_time_ns;
    double modulation_efficiency;
    
    bool is_active;
    uint64_t switching_count;
    double last_switch_time;
    double output_power_dbm;
} photonic_gate_t;

/* Photonic Processor Structure */
typedef struct {
    uint32_t num_waveguides;
    uint32_t num_gates;
    
    /* Components */
    optical_waveguide_t *waveguides;
    uint32_t waveguides_capacity;
    
    photonic_gate_t *gates;
    uint32_t gates_capacity;
    
    photon_t *photons;
    uint32_t num_photons;
    uint32_t photons_capacity;
    
    /* Photon tracking */
    uint32_t photon_counter;
    uint64_t total_photons_injected;
    uint64_t total_photons_detected;
    uint64_t total_switching_operations;
    double total_propagation_distance_mm;
    double current_time;
} photonic_processor_t;

/* Function Declarations */

/**
 * Create and initialize photonic processor
 */
photonic_processor_t* pp_create(uint32_t num_waveguides, uint32_t num_gates);

/**
 * Destroy photonic processor
 */
void pp_destroy(photonic_processor_t *pp);

/**
 * Inject photons into the optical system
 */
uint32_t pp_inject_photons(photonic_processor_t *pp, uint32_t count,
                          double power_dbm, double wavelength_nm,
                          uint32_t *photon_ids, uint32_t max_ids);

/**
 * Propagate photon through waveguide
 */
bool pp_propagate_through_waveguide(photonic_processor_t *pp, uint32_t photon_id,
                                   uint32_t waveguide_id, double *propagation_time_ns);

/**
 * Apply photonic gate operation
 */
void pp_apply_gate_operation(photonic_processor_t *pp, const uint32_t *photon_ids,
                            uint32_t count, uint32_t gate_id);

/**
 * Detect photons at output
 */
uint32_t pp_detect_photons(photonic_processor_t *pp, const uint32_t *photon_ids,
                          uint32_t count, uint32_t *detected_ids, uint32_t max_detected);

/**
 * Get processor statistics
 */
void pp_get_processor_stats(photonic_processor_t *pp, char *buffer, size_t buffer_size);

/**
 * Get gate statistics
 */
void pp_get_gate_stats(photonic_processor_t *pp, char *buffer, size_t buffer_size);

/**
 * Get waveguide statistics
 */
void pp_get_waveguide_stats(photonic_processor_t *pp, char *buffer, size_t buffer_size);

/**
 * Print processor status
 */
void pp_print_status(photonic_processor_t *pp);

#endif /* PHOTONIC_SIMULATOR_H */
