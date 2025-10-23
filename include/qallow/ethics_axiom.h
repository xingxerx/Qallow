#ifndef QALLOW_ETHICS_AXIOM_H
#define QALLOW_ETHICS_AXIOM_H

#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

#define QALLOW_ETHICS_AXIOM_TEXT "E = S + C + H"
#define QALLOW_ETHICS_COMPONENT_COUNT 3

/* Compile-time contract: the ethics axiom must always have three components. */
static_assert(QALLOW_ETHICS_COMPONENT_COUNT == 3,
              "Qallow ethics axiom requires exactly three components (S, C, H)");

typedef struct {
    double sustainability;
    double compassion;
    double harmony;
    double total;
} qallow_ethics_axiom_t;

static inline qallow_ethics_axiom_t qallow_ethics_axiom_make(double sustainability,
                                                             double compassion,
                                                             double harmony) {
    qallow_ethics_axiom_t vec = {
        .sustainability = sustainability,
        .compassion = compassion,
        .harmony = harmony,
        .total = sustainability + compassion + harmony,
    };
    return vec;
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* QALLOW_ETHICS_AXIOM_H */
