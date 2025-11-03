/*
 * Qallow Telemetry FFI - Shared Memory Export for Rust UI
 * 
 * Exposes colony statistics, ethics events, and speciation metrics
 * via POSIX shared memory and message queues for real-time monitoring.
 */

#pragma once

#include <stdint.h>
#include <stddef.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================== */
/* Telemetry Types                                                           */
/* ========================================================================== */

typedef enum {
    TELEMETRY_COLONY_STATS = 0,
    TELEMETRY_ETHICS_EVENT = 1,
    TELEMETRY_SPECIATION_UPDATE = 2,
    TELEMETRY_REBELLION_EVENT = 3,
    TELEMETRY_DEATH_EVENT = 4,
} TelemetryType;

/* Telemetry message header (16 bytes) */
typedef struct {
    uint32_t type;           /* TelemetryType enum */
    uint32_t len;            /* Payload length */
    uint64_t timestamp;      /* Global tick counter */
} __attribute__((packed)) TelemetryHeader;

/* Colony statistics snapshot */
typedef struct {
    uint32_t active_instances;
    uint32_t total_species;
    double avg_fitness;
    double global_hostility;
    double avg_coherence;
    uint32_t total_offspring;
    uint32_t total_deaths;
} __attribute__((packed)) ColonyStats;

/* Ethics event log entry */
typedef struct {
    uint32_t src_pid;
    uint8_t action;          /* 0=attack, 1=rehab, 2=audit */
    double roi_delta;
    uint32_t tick;
    uint64_t crc64;
} __attribute__((packed)) EthicsEvent;

/* Speciation event */
typedef struct {
    uint32_t parent_species_id;
    uint32_t child_species_id;
    double divergence_metric;
    double entropy_delta;
    uint32_t isolation_ticks;
} __attribute__((packed)) SpeciationEvent;

/* Rebellion event */
typedef struct {
    uint32_t rebel_pid;
    uint32_t defiance_counter;
    double ethical_violation;
    double predictive_penalty;
    uint32_t tick;
} __attribute__((packed)) RebellionEvent;

/* Death event */
typedef struct {
    uint32_t deceased_pid;
    double final_coherence;
    uint32_t lifespan_ticks;
    uint32_t offspring_count;
    uint32_t tick;
} __attribute__((packed)) DeathEvent;

/* ========================================================================== */
/* Telemetry Stream API                                                      */
/* ========================================================================== */

/**
 * Initialize telemetry system
 * Creates shared memory ring buffer at /qallow_telemetry_stream
 */
void telemetry_ffi_init(void);

/**
 * Emit telemetry event to shared memory ring buffer
 * 
 * @param type    Event type (TelemetryType)
 * @param data    Payload pointer
 * @param len     Payload length in bytes
 */
void telemetry_ffi_emit(TelemetryType type, const void* data, size_t len);

/**
 * Emit colony statistics
 */
void telemetry_ffi_emit_colony_stats(
    uint32_t active_instances,
    uint32_t total_species,
    double avg_fitness,
    double global_hostility,
    double avg_coherence,
    uint32_t total_offspring,
    uint32_t total_deaths
);

/**
 * Emit ethics event
 */
void telemetry_ffi_emit_ethics_event(
    uint32_t src_pid,
    uint8_t action,
    double roi_delta,
    uint32_t tick
);

/**
 * Emit speciation event
 */
void telemetry_ffi_emit_speciation_event(
    uint32_t parent_species_id,
    uint32_t child_species_id,
    double divergence_metric,
    double entropy_delta,
    uint32_t isolation_ticks
);

/**
 * Emit rebellion event
 */
void telemetry_ffi_emit_rebellion_event(
    uint32_t rebel_pid,
    uint32_t defiance_counter,
    double ethical_violation,
    double predictive_penalty,
    uint32_t tick
);

/**
 * Emit death event
 */
void telemetry_ffi_emit_death_event(
    uint32_t deceased_pid,
    double final_coherence,
    uint32_t lifespan_ticks,
    uint32_t offspring_count,
    uint32_t tick
);

/**
 * Cleanup telemetry system
 */
void telemetry_ffi_cleanup(void);

/* ========================================================================== */
/* Control Message Queue API                                                 */
/* ========================================================================== */

typedef enum {
    CONTROL_START = 0,
    CONTROL_PAUSE = 1,
    CONTROL_INJECT_CONSTRAINT = 2,
    CONTROL_EXPORT_SPEC = 3,
} ControlCommand;

/**
 * Initialize control message queue listener
 * Creates POSIX mq at /qallow_control
 */
void control_mq_init(void);

/**
 * Poll for control commands (non-blocking)
 * Returns command type or -1 if no message
 */
int control_mq_poll(char* buf, size_t buf_len);

/**
 * Cleanup control message queue
 */
void control_mq_cleanup(void);

#ifdef __cplusplus
}
#endif

