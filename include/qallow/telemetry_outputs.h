#ifndef QALLOW_TELEMETRY_OUTPUTS_H
#define QALLOW_TELEMETRY_OUTPUTS_H

#include <stddef.h>
#include <limits.h>

/* MSVC does not define PATH_MAX in <limits.h>; the phase engines rely on it. */
#ifndef PATH_MAX
#define PATH_MAX 260
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Resolve the CSV log path for a phase run. If `requested` is non-NULL and
 * non-empty it is used verbatim; otherwise defaults to
 * "data/logs/<phase_name>.csv". Ensures data/logs/ exists. Returns 0 on
 * success. */
int qallow_phase_resolve_log_path(const char* phase_name,
                                  const char* requested,
                                  char* out,
                                  size_t out_size);

/* Default audit tag used when the caller supplies none. */
const char* qallow_audit_tag_fallback(void);

/* Refresh "data/logs/<phase_name>_latest.csv" so tooling can always find the
 * most recent run. Symlinks require privileges on Windows, so the file is
 * copied instead. Returns 0 on success. */
int qallow_phase_update_latest_symlink(const char* phase_name,
                                       const char* csv_path);

/* Write/overwrite data/logs/phase_summary.json containing the phase name,
 * audit tag, csv path and the caller-provided metrics JSON blob. Returns 0
 * on success. */
int qallow_phase_write_summary(const char* phase_name,
                               const char* tag,
                               const char* csv_path,
                               const char* metrics_json);

#ifdef __cplusplus
}
#endif

#endif /* QALLOW_TELEMETRY_OUTPUTS_H */
