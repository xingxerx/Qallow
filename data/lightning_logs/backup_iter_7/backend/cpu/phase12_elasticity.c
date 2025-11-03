#include <math.h>
#include <stdio.h>
#include <time.h>

#include "qallow/ethics_axiom.h"
#include "qallow/telemetry_outputs.h"
#include "meta_introspect.h"

#include <limits.h>
#include <string.h>

static float clamp(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

int run_phase12_elasticity(const char* audit_tag,
                           const char* requested_log_path,
                           int ticks,
                           float eps) {
    clock_t start = clock();

    char csv_path[PATH_MAX];
    if (qallow_phase_resolve_log_path("phase12", requested_log_path, csv_path, sizeof(csv_path)) != 0) {
        fprintf(stderr, "[PHASE12] Failed to prepare log path (requested=%s)\n",
                requested_log_path ? requested_log_path : "<default>");
        return 1;
    }

    FILE* f = fopen(csv_path, "w");
    if (!f) {
        fprintf(stderr, "[PHASE12] Unable to open log file: %s\n", csv_path);
        return 1;
    }
    fprintf(f,
            "tick,coherence,entropy,decoherence,sustainability,compassion,harmony,ethics_total,audit_tag\n");

    const char* tag = (audit_tag && *audit_tag) ? audit_tag : qallow_audit_tag_fallback();

    float coherence = 0.99990f;
    float entropy   = 0.00070f;
    float deco      = 0.000009f;

    for (int t = 1; t <= ticks; ++t) {
        float stretch = clamp(eps, 0.0f, 1e-2f);
        entropy  = clamp(entropy - 0.000001f + stretch * 0.0000002f, 0.0f, 0.001f);
        coherence = clamp(1.0f - entropy * 0.2f, 0.0f, 1.0f);
        deco     = clamp(deco * (1.0f - 5e-4f) + stretch * 1e-7f, 0.0f, 0.001f);

        qallow_ethics_axiom_t ethics_vec = qallow_ethics_axiom_make(
            (double)coherence,
            (double)(1.0f - entropy),
            (double)(1.0f - deco));

        fprintf(f,
                "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,\"%s\"\n",
                t,
                coherence,
                entropy,
                deco,
                ethics_vec.sustainability,
                ethics_vec.compassion,
                ethics_vec.harmony,
                ethics_vec.total,
                tag);
    }
    fclose(f);

    printf("[PHASE12] Elastic run complete: ticks=%d eps=%.6f\n", ticks, eps);
    printf("[PHASE12] Coherence≈%.6f EntropyΔ≈%.6f Deco≈%.6f\n", coherence, entropy, deco);

    if (qallow_phase_update_latest_symlink("phase12", csv_path) != 0) {
        fprintf(stderr, "[PHASE12] Warning: failed to refresh latest symlink for %s\n", csv_path);
    }

    char metrics_json[512];
    qallow_ethics_axiom_t final_ethics = qallow_ethics_axiom_make(
        (double)coherence,
        (double)(1.0f - entropy),
        (double)(1.0f - deco));

    snprintf(metrics_json, sizeof(metrics_json),
             "{\"ticks\": %d, \"coherence\": %.6f, \"entropy\": %.6f, "
             "\"decoherence\": %.6f, \"sustainability\": %.6f, "
             "\"compassion\": %.6f, \"harmony\": %.6f, \"ethics_total\": %.6f}",
             ticks,
             coherence,
             entropy,
             deco,
             final_ethics.sustainability,
             final_ethics.compassion,
             final_ethics.harmony,
             final_ethics.total);

    if (qallow_phase_write_summary("phase12", tag, csv_path, metrics_json) != 0) {
        fprintf(stderr, "[PHASE12] Warning: failed to write phase_summary.json\n");
    }

    float duration_s = (float)(clock() - start) / (float)CLOCKS_PER_SEC;
    float coherence_metric = clamp(coherence, 0.0f, 1.0f);
    float ethics_metric = clamp(1.0f - entropy * 1000.0f, 0.0f, 1.0f);
    learn_event_t ev = {
        .phase = "phase12",
        .module = "elasticity",
        .objective_id = "phase12.elasticity",
        .duration_s = duration_s,
        .coherence = coherence_metric,
        .ethics = ethics_metric
    };
    meta_introspect_push(&ev);
    meta_introspect_flush();

    printf("[PHASE12] Artifacts:\n");
    printf("           csv=%s\n", csv_path);
    printf("           summary=data/logs/phase_summary.json\n");

    return 0;
}
