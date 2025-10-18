#include "sandbox.h"
#include <string.h>
#include <time.h>

// Sandbox/Pocket Dimension - State snapshots and rollback

CUDA_CALLABLE void sandbox_init(sandbox_manager_t* sandbox) {
    if (!sandbox) return;
    
    memset(sandbox, 0, sizeof(sandbox_manager_t));
    
    sandbox->active_snapshot_count = 0;
    sandbox->current_snapshot_index = -1;
    sandbox->isolation_active = false;
    sandbox->rollback_protection_enabled = true;
    sandbox->memory_usage = 0;
    sandbox->cpu_usage_percent = 0.0f;
    sandbox->gpu_usage_percent = 0.0f;
}

CUDA_CALLABLE bool sandbox_create_snapshot(sandbox_manager_t* sandbox, const qallow_state_t* state, const char* name) {
    if (!sandbox || !state || !name) return false;
    
    // Check if we have space
    if (sandbox->active_snapshot_count >= SANDBOX_MAX_SNAPSHOTS) {
        return false;
    }
    
    int idx = sandbox->active_snapshot_count;
    sandbox_snapshot_t* snap = &sandbox->snapshots[idx];
    
    // Copy state
    memcpy(&snap->state_snapshot, state, sizeof(qallow_state_t));
    
    // Set metadata
    strncpy(snap->name, name, SANDBOX_MAX_NAME_LENGTH - 1);
    snap->name[SANDBOX_MAX_NAME_LENGTH - 1] = '\0';
    snap->timestamp = (double)time(NULL);
    snap->is_valid = true;
    snap->safety_rating = state->global_coherence;
    
    sandbox->active_snapshot_count++;
    sandbox->current_snapshot_index = idx;
    
    return true;
}

CUDA_CALLABLE bool sandbox_load_snapshot(sandbox_manager_t* sandbox, int snapshot_index, qallow_state_t* state) {
    if (!sandbox || !state) return false;
    if (snapshot_index < 0 || snapshot_index >= sandbox->active_snapshot_count) return false;
    
    sandbox_snapshot_t* snap = &sandbox->snapshots[snapshot_index];
    if (!snap->is_valid) return false;
    
    // Restore state
    memcpy(state, &snap->state_snapshot, sizeof(qallow_state_t));
    sandbox->current_snapshot_index = snapshot_index;
    
    return true;
}

CUDA_CALLABLE bool sandbox_rollback_to_safe_state(sandbox_manager_t* sandbox, qallow_state_t* state) {
    if (!sandbox || !state) return false;
    
    // Find the safest snapshot (highest safety rating)
    int safe_idx = -1;
    float max_safety = -1.0f;
    
    for (int i = 0; i < sandbox->active_snapshot_count; i++) {
        if (sandbox->snapshots[i].is_valid && sandbox->snapshots[i].safety_rating > max_safety) {
            max_safety = sandbox->snapshots[i].safety_rating;
            safe_idx = i;
        }
    }
    
    if (safe_idx < 0) return false;
    
    return sandbox_load_snapshot(sandbox, safe_idx, state);
}

CUDA_CALLABLE void sandbox_cleanup(sandbox_manager_t* sandbox) {
    if (!sandbox) return;
    
    // Mark all snapshots as invalid
    for (int i = 0; i < SANDBOX_MAX_SNAPSHOTS; i++) {
        sandbox->snapshots[i].is_valid = false;
    }
    
    sandbox->active_snapshot_count = 0;
    sandbox->current_snapshot_index = -1;
}

CUDA_CALLABLE void sandbox_enable_isolation(sandbox_manager_t* sandbox) {
    if (!sandbox) return;
    sandbox->isolation_active = true;
}

CUDA_CALLABLE void sandbox_disable_isolation(sandbox_manager_t* sandbox) {
    if (!sandbox) return;
    sandbox->isolation_active = false;
}

CUDA_CALLABLE bool sandbox_is_state_safe(const qallow_state_t* state) {
    if (!state) return false;
    
    // State is safe if coherence is reasonable and decoherence is low
    return (state->global_coherence > 0.1f && state->decoherence_level < 0.1f);
}

void sandbox_update_resource_usage(sandbox_manager_t* sandbox) {
    if (!sandbox) return;
    
    // Calculate memory usage
    sandbox->memory_usage = sizeof(sandbox_manager_t) + 
                           (sandbox->active_snapshot_count * sizeof(sandbox_snapshot_t));
    
    // Placeholder for CPU/GPU usage
    sandbox->cpu_usage_percent = 25.0f;
    sandbox->gpu_usage_percent = 10.0f;
}

void sandbox_print_resource_report(const sandbox_manager_t* sandbox) {
    if (!sandbox) return;
    
    printf("╔════════════════════════════════════════╗\n");
    printf("║     SANDBOX RESOURCE REPORT            ║\n");
    printf("╚════════════════════════════════════════╝\n\n");
    
    printf("Active Snapshots: %d / %d\n", sandbox->active_snapshot_count, SANDBOX_MAX_SNAPSHOTS);
    printf("Memory Usage: %zu bytes\n", sandbox->memory_usage);
    printf("CPU Usage: %.1f%%\n", sandbox->cpu_usage_percent);
    printf("GPU Usage: %.1f%%\n", sandbox->gpu_usage_percent);
    printf("Isolation Active: %s\n", sandbox->isolation_active ? "YES" : "NO");
    printf("Rollback Protection: %s\n\n", sandbox->rollback_protection_enabled ? "ENABLED" : "DISABLED");
}

bool sandbox_delete_snapshot(sandbox_manager_t* sandbox, int snapshot_index) {
    if (!sandbox) return false;
    if (snapshot_index < 0 || snapshot_index >= sandbox->active_snapshot_count) return false;
    
    sandbox->snapshots[snapshot_index].is_valid = false;
    return true;
}

int sandbox_find_snapshot_by_name(const sandbox_manager_t* sandbox, const char* name) {
    if (!sandbox || !name) return -1;
    
    for (int i = 0; i < sandbox->active_snapshot_count; i++) {
        if (sandbox->snapshots[i].is_valid && 
            strcmp(sandbox->snapshots[i].name, name) == 0) {
            return i;
        }
    }
    
    return -1;
}

void sandbox_list_snapshots(const sandbox_manager_t* sandbox) {
    if (!sandbox) return;
    
    printf("╔════════════════════════════════════════╗\n");
    printf("║     SANDBOX SNAPSHOTS                  ║\n");
    printf("╚════════════════════════════════════════╝\n\n");
    
    if (sandbox->active_snapshot_count == 0) {
        printf("No snapshots available.\n\n");
        return;
    }
    
    for (int i = 0; i < sandbox->active_snapshot_count; i++) {
        const sandbox_snapshot_t* snap = &sandbox->snapshots[i];
        if (snap->is_valid) {
            printf("[%d] %s (Safety: %.4f)\n", i, snap->name, snap->safety_rating);
        }
    }
    printf("\n");
}

bool sandbox_emergency_isolation(sandbox_manager_t* sandbox, qallow_state_t* state, const char* reason) {
    if (!sandbox || !state || !reason) return false;
    
    sandbox_enable_isolation(sandbox);
    
    // Create emergency snapshot
    return sandbox_create_snapshot(sandbox, state, "emergency_isolation");
}

void sandbox_force_rollback(sandbox_manager_t* sandbox, qallow_state_t* state) {
    if (!sandbox || !state) return;
    
    sandbox_rollback_to_safe_state(sandbox, state);
}

