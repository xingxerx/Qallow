#ifndef QALLOW_ENV_H
#define QALLOW_ENV_H

#ifdef __cplusplus
extern "C" {
#endif

int qallow_env_load(const char* path);
const char* qallow_env_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* QALLOW_ENV_H */
