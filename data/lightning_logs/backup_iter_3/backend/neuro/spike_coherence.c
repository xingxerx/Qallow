#include <stdio.h>

#include "qallow/logging.h"
#include "qallow/neuro.h"

void neuromorphic_spike_demo(int nodes, float target_fidelity) {
    if (nodes <= 0) {
        nodes = 1;
    }
    if (target_fidelity <= 0.0f) {
        target_fidelity = 0.1f;
    }
    if (target_fidelity > 1.0f) {
        target_fidelity = 1.0f;
    }

    printf("[NEURO] Running spike demo on %d nodes, target=%.3f\n", nodes, target_fidelity);

    float fidelity = 0.0f;
    const float learning_rate = 0.001f;
    const int max_ticks = 2000;

    for (int tick = 0; tick < max_ticks; ++tick) {
        fidelity += learning_rate * (1.0f - fidelity);

        if ((tick % 100) == 0) {
            qallow_log_info("neuro.demo", "tick=%d fidelity=%.6f", tick, fidelity);
        }

        if (fidelity >= target_fidelity) {
            qallow_log_info("neuro.demo", "target reached tick=%d fidelity=%.6f", tick, fidelity);
            printf("[NEURO] Fidelity reached %.6f at tick %d\n", fidelity, tick);
            return;
        }
    }

    qallow_log_warn("neuro.demo",
                    "target not reached max_ticks=%d fidelity=%.6f target=%.6f",
                    max_ticks,
                    fidelity,
                    target_fidelity);
    printf("[NEURO] Max ticks reached without meeting target (final=%.6f)\n", fidelity);
}
