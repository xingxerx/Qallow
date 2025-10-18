#include <stdio.h>
#include <math.h>

static float clamp(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

int run_phase12_elasticity(const char* log_path, int ticks, float eps) {
    FILE* f = log_path ? fopen(log_path, "w") : NULL;
    if (f) fprintf(f, "tick,coherence,entropy,decoherence\n");

    float coherence = 0.99990f;
    float entropy   = 0.00070f;
    float deco      = 0.000009f;

    for (int t = 1; t <= ticks; ++t) {
        float stretch = clamp(eps, 0.0f, 1e-2f);
        entropy  = clamp(entropy - 0.000001f + stretch * 0.0000002f, 0.0f, 0.001f);
        coherence = clamp(1.0f - entropy * 0.2f, 0.0f, 1.0f);
        deco     = clamp(deco * (1.0f - 5e-4f) + stretch * 1e-7f, 0.0f, 0.001f);

        if (f) fprintf(f, "%d,%.6f,%.6f,%.6f\n", t, coherence, entropy, deco);
    }
    if (f) fclose(f);

    printf("[PHASE12] Elastic run complete: ticks=%d eps=%.6f\n", ticks, eps);
    printf("[PHASE12] Coherence≈%.6f EntropyΔ≈%.6f Deco≈%.6f\n", coherence, entropy, deco);
    return 0;
}
