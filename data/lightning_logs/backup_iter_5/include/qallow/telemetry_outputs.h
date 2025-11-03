#ifndef QALLOW_TELEMETRY_OUTPUTS_H
#define QALLOW_TELEMETRY_OUTPUTS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Resolve the CSV log path for a phase, creating directories as needed. */
int qallow_phase_resolve_log_path(const char* phase,
                                  const char* override_path,
                                  char* out_path,
                                  size_t out_len);

/* Update the rolling latest/ symlink for a phase CSV file. */
int qallow_phase_update_latest_symlink(const char* phase, const char* csv_path);

/* Write the phase_summary.json payload with phase metadata and metrics. */
int qallow_phase_write_summary(const char* phase,
                               const char* audit_tag,
                               const char* csv_path,
                               const char* metrics_json);

/* Resolve the default audit tag from CLI/environment fallbacks. */
const char* qallow_audit_tag_fallback(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* QALLOW_TELEMETRY_OUTPUTS_H */
