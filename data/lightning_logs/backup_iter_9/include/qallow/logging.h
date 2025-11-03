#ifndef QALLOW_LOGGING_H
#define QALLOW_LOGGING_H

#ifdef __cplusplus
extern "C" {
#endif

void qallow_logging_init(void);
void qallow_logging_shutdown(void);
void qallow_logging_set_directory(const char* dir);
void qallow_logging_flush(void);

void qallow_log_info(const char* scope, const char* fmt, ...);
void qallow_log_warn(const char* scope, const char* fmt, ...);
void qallow_log_error(const char* scope, const char* fmt, ...);

#ifdef __cplusplus
}
#endif

#endif /* QALLOW_LOGGING_H */
