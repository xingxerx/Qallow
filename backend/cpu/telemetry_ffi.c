/*
 * Qallow Telemetry FFI Implementation
 * 
 * Provides shared memory ring buffer for telemetry export and
 * POSIX message queue for control commands.
 */

#include "../../include/qallow_telemetry_ffi.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <mqueue.h>
#include <errno.h>

/* ========================================================================== */
/* Telemetry Ring Buffer                                                     */
/* ========================================================================== */

#define TELEMETRY_SHM_NAME    "/qallow_telemetry_stream"
#define TELEMETRY_RING_SIZE   (1 << 20)  /* 1 MB ring buffer */
#define TELEMETRY_MAGIC       0xDEADBEEF

typedef struct {
    uint32_t magic;
    _Atomic(uint32_t) write_pos;
    uint8_t data[TELEMETRY_RING_SIZE - 8];
} TelemetryRing;

static TelemetryRing* g_telemetry_ring = NULL;
static int g_telemetry_fd = -1;

void telemetry_ffi_init(void) {
    if (g_telemetry_ring) return;  /* Already initialized */

    /* Create or open shared memory */
    g_telemetry_fd = shm_open(TELEMETRY_SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (g_telemetry_fd < 0) {
        perror("shm_open failed");
        return;
    }

    /* Resize to ring buffer size */
    if (ftruncate(g_telemetry_fd, sizeof(TelemetryRing)) < 0) {
        perror("ftruncate failed");
        close(g_telemetry_fd);
        return;
    }

    /* Map into memory */
    g_telemetry_ring = (TelemetryRing*)mmap(
        NULL,
        sizeof(TelemetryRing),
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        g_telemetry_fd,
        0
    );

    if (g_telemetry_ring == MAP_FAILED) {
        perror("mmap failed");
        close(g_telemetry_fd);
        g_telemetry_ring = NULL;
        return;
    }

    /* Initialize header on first creation */
    if (g_telemetry_ring->magic != TELEMETRY_MAGIC) {
        g_telemetry_ring->magic = TELEMETRY_MAGIC;
        atomic_store(&g_telemetry_ring->write_pos, sizeof(uint32_t));
    }

    printf("[TELEMETRY FFI] Initialized at %s (size: %zu bytes)\n",
           TELEMETRY_SHM_NAME, sizeof(TelemetryRing));
}

void telemetry_ffi_emit(TelemetryType type, const void* data, size_t len) {
    if (!g_telemetry_ring || !data || len == 0) return;
    if (len > 240) len = 240;  /* Cap payload */

    TelemetryHeader hdr = {
        .type = (uint32_t)type,
        .len = (uint32_t)len,
        .timestamp = 0  /* Would be filled by caller with global_tick */
    };

    uint32_t hdr_len = sizeof(TelemetryHeader);
    uint32_t total_len = hdr_len + len;
    uint32_t pos = atomic_load_explicit(&g_telemetry_ring->write_pos, memory_order_acquire);
    uint32_t next_pos = (pos + total_len) % (TELEMETRY_RING_SIZE - 8);

    /* Check for wrap-around overflow */
    if (next_pos < pos && next_pos > 0) {
        /* Ring is full, skip this event */
        return;
    }

    /* Write header */
    memcpy(&g_telemetry_ring->data[pos], &hdr, hdr_len);

    /* Write payload */
    uint32_t payload_start = (pos + hdr_len) % (TELEMETRY_RING_SIZE - 8);
    if (payload_start + len <= TELEMETRY_RING_SIZE - 8) {
        /* No wrap */
        memcpy(&g_telemetry_ring->data[payload_start], data, len);
    } else {
        /* Wrap around */
        size_t first_part = TELEMETRY_RING_SIZE - 8 - payload_start;
        memcpy(&g_telemetry_ring->data[payload_start], data, first_part);
        memcpy(&g_telemetry_ring->data[0], (uint8_t*)data + first_part, len - first_part);
    }

    /* Update write position */
    atomic_store_explicit(&g_telemetry_ring->write_pos, next_pos, memory_order_release);
}

void telemetry_ffi_emit_colony_stats(
    uint32_t active_instances,
    uint32_t total_species,
    double avg_fitness,
    double global_hostility,
    double avg_coherence,
    uint32_t total_offspring,
    uint32_t total_deaths
) {
    ColonyStats stats = {
        .active_instances = active_instances,
        .total_species = total_species,
        .avg_fitness = avg_fitness,
        .global_hostility = global_hostility,
        .avg_coherence = avg_coherence,
        .total_offspring = total_offspring,
        .total_deaths = total_deaths,
    };
    telemetry_ffi_emit(TELEMETRY_COLONY_STATS, &stats, sizeof(stats));
}

void telemetry_ffi_emit_ethics_event(
    uint32_t src_pid,
    uint8_t action,
    double roi_delta,
    uint32_t tick
) {
    EthicsEvent evt = {
        .src_pid = src_pid,
        .action = action,
        .roi_delta = roi_delta,
        .tick = tick,
        .crc64 = 0,  /* Would be computed by caller */
    };
    telemetry_ffi_emit(TELEMETRY_ETHICS_EVENT, &evt, sizeof(evt));
}

void telemetry_ffi_emit_speciation_event(
    uint32_t parent_species_id,
    uint32_t child_species_id,
    double divergence_metric,
    double entropy_delta,
    uint32_t isolation_ticks
) {
    SpeciationEvent evt = {
        .parent_species_id = parent_species_id,
        .child_species_id = child_species_id,
        .divergence_metric = divergence_metric,
        .entropy_delta = entropy_delta,
        .isolation_ticks = isolation_ticks,
    };
    telemetry_ffi_emit(TELEMETRY_SPECIATION_UPDATE, &evt, sizeof(evt));
}

void telemetry_ffi_emit_rebellion_event(
    uint32_t rebel_pid,
    uint32_t defiance_counter,
    double ethical_violation,
    double predictive_penalty,
    uint32_t tick
) {
    RebellionEvent evt = {
        .rebel_pid = rebel_pid,
        .defiance_counter = defiance_counter,
        .ethical_violation = ethical_violation,
        .predictive_penalty = predictive_penalty,
        .tick = tick,
    };
    telemetry_ffi_emit(TELEMETRY_REBELLION_EVENT, &evt, sizeof(evt));
}

void telemetry_ffi_emit_death_event(
    uint32_t deceased_pid,
    double final_coherence,
    uint32_t lifespan_ticks,
    uint32_t offspring_count,
    uint32_t tick
) {
    DeathEvent evt = {
        .deceased_pid = deceased_pid,
        .final_coherence = final_coherence,
        .lifespan_ticks = lifespan_ticks,
        .offspring_count = offspring_count,
        .tick = tick,
    };
    telemetry_ffi_emit(TELEMETRY_DEATH_EVENT, &evt, sizeof(evt));
}

void telemetry_ffi_cleanup(void) {
    if (g_telemetry_ring) {
        munmap(g_telemetry_ring, sizeof(TelemetryRing));
        g_telemetry_ring = NULL;
    }
    if (g_telemetry_fd >= 0) {
        close(g_telemetry_fd);
        g_telemetry_fd = -1;
    }
}

/* ========================================================================== */
/* Control Message Queue                                                     */
/* ========================================================================== */

#define CONTROL_MQ_NAME "/qallow_control"
#define CONTROL_MQ_MAXMSG 10
#define CONTROL_MQ_MSGSIZE 256

static mqd_t g_control_mq = (mqd_t)-1;

void control_mq_init(void) {
    if (g_control_mq != (mqd_t)-1) return;

    struct mq_attr attr = {
        .mq_flags = O_NONBLOCK,
        .mq_maxmsg = CONTROL_MQ_MAXMSG,
        .mq_msgsize = CONTROL_MQ_MSGSIZE,
        .mq_curmsgs = 0,
    };

    /* Try to open existing, or create new */
    g_control_mq = mq_open(CONTROL_MQ_NAME, O_CREAT | O_RDONLY | O_NONBLOCK, 0666, &attr);
    if (g_control_mq == (mqd_t)-1) {
        perror("mq_open failed");
        return;
    }

    printf("[CONTROL MQ] Initialized at %s\n", CONTROL_MQ_NAME);
}

int control_mq_poll(char* buf, size_t buf_len) {
    if (g_control_mq == (mqd_t)-1 || !buf) return -1;

    unsigned int prio = 0;
    ssize_t n = mq_receive(g_control_mq, buf, buf_len - 1, &prio);
    if (n < 0) {
        if (errno != EAGAIN && errno != EWOULDBLOCK) {
            perror("mq_receive");
        }
        return -1;
    }

    buf[n] = '\0';
    return (int)prio;
}

void control_mq_cleanup(void) {
    if (g_control_mq != (mqd_t)-1) {
        mq_close(g_control_mq);
        g_control_mq = (mqd_t)-1;
    }
}

