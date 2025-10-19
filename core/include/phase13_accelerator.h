#ifndef QALLOW_PHASE13_ACCELERATOR_H
#define QALLOW_PHASE13_ACCELERATOR_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Configuration passed to the Phase-13 accelerator runtime.
 * - thread_count: 0 selects the number of online CPUs automatically.
 * - watch_dir: directory to monitor via inotify (NULL to disable watching).
 * - files: optional list of files to enqueue for immediate processing.
 * - file_count: number of entries in the files array.
 * - keep_running: non-zero to keep the service loop alive even when no watch directory is specified.
 */
typedef struct phase13_accel_config_s {
    size_t thread_count;
    const char* watch_dir;
    const char* const* files;
    size_t file_count;
    int keep_running;
    int remote_sync_enabled;
    const char* remote_sync_endpoint;
    unsigned int remote_sync_interval_sec;
} phase13_accel_config_t;

/**
 * Parse CLI arguments and launch the accelerator. Mirrors the behaviour of the standalone binary.
 */
int qallow_phase13_main(int argc, char** argv);

/**
 * Launch the accelerator with a programmatic configuration.
 */
int qallow_phase13_accel_start(const phase13_accel_config_t* config);

#ifdef __cplusplus
}
#endif

#endif /* QALLOW_PHASE13_ACCELERATOR_H */
