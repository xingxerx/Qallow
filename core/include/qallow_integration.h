#ifndef QALLOW_INTEGRATION_H
#define QALLOW_INTEGRATION_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    QALLOW_LATTICE_PHASE14 = 1u << 0,
    QALLOW_LATTICE_PHASE15 = 1u << 1
} qallow_lattice_phase_t;

typedef struct {
    uint32_t phase_mask;
    int ticks;
    bool no_split;
    bool print_summary;
} qallow_lattice_config_t;

void qallow_lattice_config_init(qallow_lattice_config_t* cfg);
void qallow_lattice_config_enable(qallow_lattice_config_t* cfg, qallow_lattice_phase_t phase, bool enable);

int qallow_lattice_integrate(const qallow_lattice_config_t* cfg);

#ifdef __cplusplus
}
#endif

#endif /* QALLOW_INTEGRATION_H */
