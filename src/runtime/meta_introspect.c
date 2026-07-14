#include "meta_introspect.h"

#include <stdio.h>
#include <time.h>

#ifdef _WIN32
#include <direct.h>
#define qallow_mi_mkdir(path) _mkdir(path)
#else
#include <sys/stat.h>
#include <sys/types.h>
#define qallow_mi_mkdir(path) mkdir(path, 0755)
#endif

void meta_introspect_push(const learn_event_t* ev) {
    if (!ev) {
        return;
    }
    (void)qallow_mi_mkdir("data");
    (void)qallow_mi_mkdir("data/logs");

    FILE* f = fopen("data/logs/meta_introspect.jsonl", "a");
    if (!f) {
        return;
    }
    fprintf(f,
            "{\"ts\": %lld, \"phase\": \"%s\", \"module\": \"%s\", "
            "\"objective_id\": \"%s\", \"duration_s\": %.6f, "
            "\"coherence\": %.6f, \"ethics\": %.6f}\n",
            (long long)time(NULL),
            ev->phase ? ev->phase : "",
            ev->module ? ev->module : "",
            ev->objective_id ? ev->objective_id : "",
            ev->duration_s,
            ev->coherence,
            ev->ethics);
    fclose(f);
}

void meta_introspect_flush(void) {
    /* Events are written directly on push; nothing buffered. */
}
