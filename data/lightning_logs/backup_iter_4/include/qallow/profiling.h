#ifndef QALLOW_PROFILING_H
#define QALLOW_PROFILING_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char* name;
    double start_ns;
    int active;
} qallow_profile_scope_t;

qallow_profile_scope_t qallow_profile_scope_enter(const char* name);
void qallow_profile_scope_exit(qallow_profile_scope_t* scope);

#ifdef __cplusplus
}
#endif

#define QALLOW_PROFILE_SCOPE(name)                                                       \
    for (qallow_profile_scope_t qallow__scope__ = qallow_profile_scope_enter(name);       \
         qallow__scope__.active;                                                          \
         qallow_profile_scope_exit(&qallow__scope__))

#endif /* QALLOW_PROFILING_H */
