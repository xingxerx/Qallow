#ifndef QALLOW_ETHICS_AXIOM_H
#define QALLOW_ETHICS_AXIOM_H

/* Ethics axiom vector used by the phase engines.
 *
 * Formula (matches the README's ethics score E = S + C + H):
 *   sustainability = clamp01(a)   -- callers pass coherence
 *   compassion     = clamp01(b)   -- callers pass 1 - entropy (or 1 - drift)
 *   harmony        = clamp01(c)   -- callers pass 1 - decoherence (or 1 - energy)
 *   total          = S + C + H    -- in [0, 3]
 */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double sustainability;
    double compassion;
    double harmony;
    double total;
} qallow_ethics_axiom_t;

static double qallow_ethics_clamp01_(double v) {
    if (v < 0.0) return 0.0;
    if (v > 1.0) return 1.0;
    return v;
}

static qallow_ethics_axiom_t qallow_ethics_axiom_make(double a, double b, double c) {
    qallow_ethics_axiom_t out;
    out.sustainability = qallow_ethics_clamp01_(a);
    out.compassion     = qallow_ethics_clamp01_(b);
    out.harmony        = qallow_ethics_clamp01_(c);
    out.total          = out.sustainability + out.compassion + out.harmony;
    return out;
}

#ifdef __cplusplus
}
#endif

#endif /* QALLOW_ETHICS_AXIOM_H */
