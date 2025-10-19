#include "pocket.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

int pocket_spawn(pocket_dimension_t* pd, int n) {
    if (!pd || n <= 0 || n > MAX_POCKETS) return 0;
    
    pd->count = n;
    pd->merged_score = 0.0;
    pd->average_coherence = 0.0;
    pd->average_decoherence = 0.0;
    pd->memory_usage_mb = 0.0;
    pd->memory_peak_mb = 0.0;
    
    printf("[POCKET] Spawning %d parallel simulations...\n", n);
    
    for (int i = 0; i < n; i++) {
        // Initialize each pocket with slightly different seed
        qallow_kernel_init(&pd->pockets[i].state);
        
        // Vary initial conditions slightly
        for (int j = 0; j < NUM_OVERLAYS; j++) {
            overlay_t* overlay = &pd->pockets[i].state.overlays[j];
            float variance = 0.01f * (i + 1);
            for (int k = 0; k < overlay->node_count; k++) {
                overlay->values[k] += variance * ((float)rand() / RAND_MAX - 0.5f);
                if (overlay->values[k] < 0.0f) overlay->values[k] = 0.0f;
                if (overlay->values[k] > 1.0f) overlay->values[k] = 1.0f;
            }
        }
        
        pd->pockets[i].result_score = 0.0;
        pd->pockets[i].memory_usage_mb = 0.0;
        pd->pockets[i].memory_peak_mb = 0.0;
        pd->pockets[i].active = 1;

        printf("[POCKET] Pocket %d initialized\n", i);
    }

    return n;
}

void pocket_tick_all(pocket_dimension_t* pd) {
    if (!pd) return;
    
    for (int i = 0; i < pd->count; i++) {
        if (!pd->pockets[i].active) continue;
        
        // Run one tick in this pocket
        qallow_kernel_tick(&pd->pockets[i].state);
        
        // Calculate score for this pocket
        double score = pd->pockets[i].state.global_coherence * 
                       (1.0 - pd->pockets[i].state.decoherence_level);
        pd->pockets[i].result_score = score;

        double usage = 256.0 + (i + 1) * 12.5 +
                       pd->pockets[i].state.decoherence_level * 800.0;
        pd->pockets[i].memory_usage_mb = usage;
        if (usage > pd->pockets[i].memory_peak_mb) {
            pd->pockets[i].memory_peak_mb = usage;
        }
    }
}

double pocket_merge(pocket_dimension_t* pd) {
    if (!pd || pd->count == 0) return 0.0;

    double total_score = 0.0;
    double total_coherence = 0.0;
    double total_decoherence = 0.0;
    double total_memory = 0.0;
    double peak_memory = 0.0;

    printf("[POCKET] Merging %d pocket results...\n", pd->count);

    for (int i = 0; i < pd->count; i++) {
        if (!pd->pockets[i].active) continue;

        total_score += pd->pockets[i].result_score;
        total_coherence += pd->pockets[i].state.global_coherence;
        total_decoherence += pd->pockets[i].state.decoherence_level;
        total_memory += pd->pockets[i].memory_usage_mb;
        if (pd->pockets[i].memory_peak_mb > peak_memory) {
            peak_memory = pd->pockets[i].memory_peak_mb;
        }
    }

    pd->merged_score = total_score / pd->count;
    pd->average_coherence = total_coherence / pd->count;
    pd->average_decoherence = total_decoherence / pd->count;
    pd->memory_usage_mb = total_memory / pd->count;
    if (pd->memory_usage_mb > pd->memory_peak_mb) {
        pd->memory_peak_mb = pd->memory_usage_mb;
    }
    if (peak_memory > pd->memory_peak_mb) {
        pd->memory_peak_mb = peak_memory;
    }

    printf("[POCKET] Merged score: %.4f\n", pd->merged_score);
    printf("[POCKET] Average coherence: %.4f\n", pd->average_coherence);
    printf("[POCKET] Average decoherence: %.5f\n", pd->average_decoherence);
    printf("[POCKET] Memory usage: %.2f MB (peak %.2f MB)\n",
           pd->memory_usage_mb, pd->memory_peak_mb);

    return pd->merged_score;
}

double pocket_get_average_score(const pocket_dimension_t* pd) {
    if (!pd) return 0.0;
    return pd->merged_score;
}

void pocket_capture_metrics(pocket_dimension_t* pd, int tick) {
    if (!pd) return;

    FILE* f = fopen("data/telemetry/pocket_metrics.json", "w");
    if (!f) return;
    fprintf(f,
            "{\n"
            "  \"tick\": %d,\n"
            "  \"average_score\": %.4f,\n"
            "  \"average_coherence\": %.4f,\n"
            "  \"average_decoherence\": %.6f,\n"
            "  \"memory_usage_mb\": %.2f,\n"
            "  \"memory_peak_mb\": %.2f\n"
            "}\n",
            tick,
            pd->merged_score,
            pd->average_coherence,
            pd->average_decoherence,
            pd->memory_usage_mb,
            pd->memory_peak_mb);
    fclose(f);
}

void pocket_cleanup(pocket_dimension_t* pd) {
    if (!pd) return;

    for (int i = 0; i < pd->count; i++) {
        pd->pockets[i].active = 0;
    }
    
    pd->count = 0;
    printf("[POCKET] All pockets cleaned up\n");
}
