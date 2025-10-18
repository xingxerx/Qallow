#ifndef SANDBOX_H
#define SANDBOX_H

#include "qallow_kernel.h"

// Sandbox/Pocket Dimension module
// Handles state snapshots, isolation, and rollback capabilities

#define SANDBOX_MAX_SNAPSHOTS 10
#define SANDBOX_MAX_NAME_LENGTH 64

typedef struct {
    char name[SANDBOX_MAX_NAME_LENGTH];
    qallow_state_t state_snapshot;
    double timestamp;
    bool is_valid;
    float safety_rating;
} sandbox_snapshot_t;

typedef struct {
    sandbox_snapshot_t snapshots[SANDBOX_MAX_SNAPSHOTS];
    int active_snapshot_count;
    int current_snapshot_index;
    bool isolation_active;
    bool rollback_protection_enabled;
    
    // Resource tracking
    size_t memory_usage;
    float cpu_usage_percent;
    float gpu_usage_percent;
} sandbox_manager_t;

// Function declarations
CUDA_CALLABLE void sandbox_init(sandbox_manager_t* sandbox);
CUDA_CALLABLE bool sandbox_create_snapshot(sandbox_manager_t* sandbox, const qallow_state_t* state, const char* name);
CUDA_CALLABLE bool sandbox_load_snapshot(sandbox_manager_t* sandbox, int snapshot_index, qallow_state_t* state);
CUDA_CALLABLE bool sandbox_rollback_to_safe_state(sandbox_manager_t* sandbox, qallow_state_t* state);
CUDA_CALLABLE void sandbox_cleanup(sandbox_manager_t* sandbox);

// Isolation functions
CUDA_CALLABLE void sandbox_enable_isolation(sandbox_manager_t* sandbox);
CUDA_CALLABLE void sandbox_disable_isolation(sandbox_manager_t* sandbox);
CUDA_CALLABLE bool sandbox_is_state_safe(const qallow_state_t* state);

// Resource monitoring
void sandbox_update_resource_usage(sandbox_manager_t* sandbox);
void sandbox_print_resource_report(const sandbox_manager_t* sandbox);

// Snapshot management
bool sandbox_delete_snapshot(sandbox_manager_t* sandbox, int snapshot_index);
int sandbox_find_snapshot_by_name(const sandbox_manager_t* sandbox, const char* name);
void sandbox_list_snapshots(const sandbox_manager_t* sandbox);

// Emergency procedures
bool sandbox_emergency_isolation(sandbox_manager_t* sandbox, qallow_state_t* state, const char* reason);
void sandbox_force_rollback(sandbox_manager_t* sandbox, qallow_state_t* state);

#endif // SANDBOX_H