#ifndef QALLOW_META_INTROSPECT_H
#define QALLOW_META_INTROSPECT_H

#ifdef __cplusplus
extern "C" {
#endif

/* Lightweight learning-event record emitted at the end of a phase run. */
typedef struct {
    const char* phase;
    const char* module;
    const char* objective_id;
    float duration_s;
    float coherence;
    float ethics;
} learn_event_t;

/* Append the event as a JSON line to data/logs/meta_introspect.jsonl. */
void meta_introspect_push(const learn_event_t* ev);

/* Flush any buffered introspection output. */
void meta_introspect_flush(void);

#ifdef __cplusplus
}
#endif

#endif /* QALLOW_META_INTROSPECT_H */
