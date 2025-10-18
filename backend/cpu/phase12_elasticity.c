#include <stdio.h>
#include <math.h>
#include "qallow_phase12.h"

static float clamp(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

int run_phase12_elasticity(const char* log_path, int ticks, float eps) {
    FILE* f = log_path ? fopen(log_path, "w") : NULL;
    if (f) {
        fprintf(f, "tick,coherence,entropy,decoherence\n");
    }
    
    float coherence = 0.99990f;
    float entropy   = 0.00070f;
    float deco      = 0.000009f;

    printf("[PHASE12] Starting elastic simulation...\n");
    printf("[PHASE12] Parameters: ticks=%d eps=%.6f\n", ticks, eps);
    if (log_path) {
        printf("[PHASE12] Logging to: %s\n", log_path);
    }
    printf("\n");

    for (int t = 1; t <= ticks; ++t) {
        // Elastic extension without collapse: Ψ' = Ψ ⊗ (I+ε)
        float stretch = clamp(eps, 0.0f, 1e-2f);  // ≤0.01% limit
        
        // Update dynamics with elastic perturbation
        entropy   = clamp(entropy - 0.000001f + stretch * 0.0000002f, 0.0f, 0.001f);
        coherence = clamp(1.0f - entropy * 0.2f, 0.0f, 1.0f);
        deco      = clamp(deco * (1.0f - 5e-4f) + stretch * 1e-7f, 0.0f, 0.001f);

        // Log to CSV
        if (f) {
            fprintf(f, "%d,%.6f,%.6f,%.6f\n", t, coherence, entropy, deco);
        }

        // Progress indicator every 100 ticks
        if (t % 100 == 0) {
            printf("[PHASE12] Tick %d/%d | Coherence=%.6f Entropy=%.6f Deco=%.6f\n",
                   t, ticks, coherence, entropy, deco);
        }
    }
    
    if (f) {
        fclose(f);
    }

    printf("\n");
    printf("[PHASE12] ═══════════════════════════════════════════════════\n");
    printf("[PHASE12] Elastic run complete: ticks=%d eps=%.6f\n", ticks, eps);
    printf("[PHASE12] Final State:\n");
    printf("[PHASE12]   Coherence  ≈ %.6f\n", coherence);
    printf("[PHASE12]   Entropy Δ  ≈ %.6f\n", entropy);
    printf("[PHASE12]   Decoherence≈ %.6f\n", deco);
    printf("[PHASE12] ═══════════════════════════════════════════════════\n");
    
    return 0;
}
