#define _GNU_SOURCE
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/inotify.h>
#include <unistd.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <stdarg.h>
#include <errno.h>
#include <limits.h>
#include <time.h>
#include <stdbool.h>

#ifdef __has_include
#  if __has_include("phase13_accelerator.h")
#    include "phase13_accelerator.h"
#  elif __has_include("../core/include/phase13_accelerator.h")
#    include "../core/include/phase13_accelerator.h"
#  else
#    error "phase13_accelerator.h not found"
#  endif
#else
#  include "phase13_accelerator.h"
#endif

#ifndef QALLOW_CACHE_ENTRIES
#define QALLOW_CACHE_ENTRIES 2048
#endif
#define QALLOW_KEY_MAX 1280
#define QALLOW_VAL_MAX 256
#define QALLOW_SHM_NAME "/qallow_cache_v1"
#define QALLOW_EVENT_BUFSZ (64 * 1024)

typedef struct {
    atomic_uint_least64_t tag;     // 0 = empty, else hash
    char key[QALLOW_KEY_MAX];      // path|mtime
    char val[QALLOW_VAL_MAX];      // cached “analysis”
} cache_entry_t;

typedef struct {
    cache_entry_t slots[QALLOW_CACHE_ENTRIES];
} cache_t;

typedef struct job_s {
    char path[1024];
    time_t mtime;
} job_t;

typedef struct node_s {
    job_t job;
    struct node_s* next;
} node_t;

static cache_t* g_cache = NULL;
static size_t g_threads = 0;
static node_t* q_head = NULL;
static node_t* q_tail = NULL;
static pthread_mutex_t q_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t q_cv = PTHREAD_COND_INITIALIZER;
static atomic_int stop_flag = 0;
static atomic_uint pending_jobs = 0;
static atomic_uint remote_sync_seq = 0;
static int g_quiet = 0;

static void queue_push(job_t j);

typedef struct remote_sync_state_s {
    int enabled;
    int thread_started;
    pthread_t thread;
    char endpoint[512];
    char target_dir[PATH_MAX];
    unsigned int interval_sec;
} remote_sync_state_t;

static void qallow_infof(const char* fmt, ...) {
    if (g_quiet) {
        return;
    }
    va_list args;
    va_start(args, fmt);
    vfprintf(stdout, fmt, args);
    va_end(args);
}

static int ensure_directory(const char* path) {
    if (!path || !*path) {
        return -1;
    }

    char tmp[PATH_MAX];
    snprintf(tmp, sizeof(tmp), "%s", path);

    size_t len = strlen(tmp);
    if (len == 0) {
        return -1;
    }

    // Trim trailing slash to avoid double-creating
    if (tmp[len - 1] == '/') {
        tmp[len - 1] = '\0';
    }

    for (char* p = tmp + 1; *p; ++p) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(tmp, 0755) < 0 && errno != EEXIST) {
                *p = '/';
                return -1;
            }
            *p = '/';
        }
    }

    if (mkdir(tmp, 0755) < 0 && errno != EEXIST) {
        return -1;
    }

    return 0;
}

static void* remote_sync_thread(void* arg) {
    remote_sync_state_t* state = (remote_sync_state_t*)arg;
   if (!state) {
        return NULL;
    }

    qallow_infof("[REMOTE_SYNC] Remote synchronization active (endpoint=%s, target=%s, interval=%u s)\n",
                 state->endpoint, state->target_dir, state->interval_sec);

    while (!atomic_load_explicit(&stop_flag, memory_order_acquire)) {
        unsigned int seq = atomic_fetch_add_explicit(&remote_sync_seq, 1, memory_order_relaxed) + 1;
        char filepath[PATH_MAX];
        int path_len = snprintf(filepath, sizeof(filepath), "%s/remote_batch_%u.json",
                                state->target_dir, seq);
        if (path_len < 0 || (size_t)path_len >= sizeof(filepath)) {
            fprintf(stderr, "[REMOTE_SYNC] Remote path truncated for base %s\n", state->target_dir);
            continue;
        }

        FILE* f = fopen(filepath, "w");
        if (f) {
            time_t now = time(NULL);
            fprintf(f,
                    "{ \"id\": %u, \"timestamp\": %ld, \"endpoint\": \"%s\", \"hint\": \"synthesized\" }\n",
                    seq, (long)now, state->endpoint);
            fclose(f);

            job_t job;
            memset(&job, 0, sizeof(job));
            int job_path_len = snprintf(job.path, sizeof(job.path), "%s", filepath);
            if (job_path_len < 0 || (size_t)job_path_len >= sizeof(job.path)) {
                fprintf(stderr, "[REMOTE_SYNC] Job path truncated: %s\n", filepath);
                continue;
            }
            job.mtime = now;
            queue_push(job);

            qallow_infof("[REMOTE_SYNC] Enqueued remote artifact %s\n", filepath);
        } else {
            fprintf(stderr, "[REMOTE_SYNC] Failed to write %s: %s\n", filepath, strerror(errno));
        }

        unsigned int slices = state->interval_sec * 5;
        if (slices == 0) {
            slices = 5;
        }
        const struct timespec ts = {.tv_sec = 0, .tv_nsec = 200 * 1000 * 1000};
        for (unsigned int step = 0; step < slices; ++step) {
            if (atomic_load_explicit(&stop_flag, memory_order_acquire)) {
                break;
            }
            nanosleep(&ts, NULL);
        }
    }

    qallow_infof("[REMOTE_SYNC] Remote synchronization loop stopped\n");
    return NULL;
}

static void remote_sync_prepare(const phase13_accel_config_t* cfg, remote_sync_state_t* state) {
    if (!cfg || !state) {
        return;
    }

    memset(state, 0, sizeof(*state));

    if (!cfg->remote_sync_enabled) {
        return;
    }

    state->enabled = 1;

    const char* endpoint = cfg->remote_sync_endpoint;
    if (!endpoint || !*endpoint) {
        endpoint = getenv("QALLOW_REMOTE_SYNC_ENDPOINT");
    }
    if (!endpoint || !*endpoint) {
        endpoint = "remote://default";
    }
    snprintf(state->endpoint, sizeof(state->endpoint), "%s", endpoint);

    const char* base_dir = cfg->watch_dir;
    if (!base_dir || !*base_dir) {
        base_dir = getenv("QALLOW_REMOTE_SYNC_DIR");
    }
    if (!base_dir || !*base_dir) {
        base_dir = "./remote-sync";
    }
    snprintf(state->target_dir, sizeof(state->target_dir), "%s", base_dir);

    if (ensure_directory(state->target_dir) != 0) {
        fprintf(stderr, "[REMOTE_SYNC] Unable to prepare directory: %s\n", state->target_dir);
        state->enabled = 0;
        return;
    }

    unsigned int interval = cfg->remote_sync_interval_sec;
    if (interval == 0) {
        const char* env_interval = getenv("QALLOW_REMOTE_SYNC_INTERVAL");
        if (env_interval && *env_interval) {
            char* end = NULL;
            long v = strtol(env_interval, &end, 10);
            if (end && *end == '\0' && v > 0) {
                interval = (unsigned int)v;
            }
        }
    }
    if (interval == 0) {
        interval = 30;
    }
    state->interval_sec = interval;
}

static void remote_sync_start(remote_sync_state_t* state) {
    if (!state || !state->enabled) {
        return;
    }

    if (pthread_create(&state->thread, NULL, remote_sync_thread, state) != 0) {
        fprintf(stderr, "[REMOTE_SYNC] Failed to create sync thread: %s\n", strerror(errno));
        state->thread_started = 0;
        state->enabled = 0;
        return;
    }

    state->thread_started = 1;
}

static void remote_sync_stop(remote_sync_state_t* state) {
    if (!state || !state->thread_started) {
        return;
    }

    pthread_join(state->thread, NULL);
    state->thread_started = 0;
}

static uint64_t fnv1a64(const void* data, size_t len) {
    const uint8_t* p = (const uint8_t*)data;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static cache_t* cache_attach(void) {
    int fd = shm_open(QALLOW_SHM_NAME, O_CREAT | O_RDWR, 0600);
    if (fd < 0) {
        perror("shm_open");
        return NULL;
    }
    size_t sz = sizeof(cache_t);
    if (ftruncate(fd, sz) < 0) {
        perror("ftruncate");
        close(fd);
        return NULL;
    }
    void* p = mmap(NULL, sz, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return NULL;
    }
    close(fd);
    return (cache_t*)p;
}

// linear-probe insert/find
static int cache_get(const char* key, char* out, size_t out_sz) {
    uint64_t h = fnv1a64(key, strlen(key));
    size_t idx = h % QALLOW_CACHE_ENTRIES;
    for (size_t i = 0; i < QALLOW_CACHE_ENTRIES; i++) {
        size_t j = (idx + i) % QALLOW_CACHE_ENTRIES;
        uint64_t tag = atomic_load_explicit(&g_cache->slots[j].tag, memory_order_acquire);
        if (tag == 0) return 0; // stop at empty
        if (tag == h && strncmp(g_cache->slots[j].key, key, QALLOW_KEY_MAX) == 0) {
            if (out_sz > 0) {
                snprintf(out, out_sz, "%s", g_cache->slots[j].val);
            }
            return 1;
        }
    }
    return 0;
}

static void cache_put(const char* key, const char* val) {
    uint64_t h = fnv1a64(key, strlen(key));
    size_t idx = h % QALLOW_CACHE_ENTRIES;
    for (size_t i = 0; i < QALLOW_CACHE_ENTRIES; i++) {
        size_t j = (idx + i) % QALLOW_CACHE_ENTRIES;
    uint64_t expect = 0;
    if (atomic_compare_exchange_strong(&g_cache->slots[j].tag, &expect, h)) {
            snprintf(g_cache->slots[j].key, QALLOW_KEY_MAX, "%s", key);
            snprintf(g_cache->slots[j].val, QALLOW_VAL_MAX, "%s", val);
            atomic_thread_fence(memory_order_release);
            return;
        }
        if (expect == h && strncmp(g_cache->slots[j].key, key, QALLOW_KEY_MAX) == 0) {
            snprintf(g_cache->slots[j].val, QALLOW_VAL_MAX, "%s", val);
            atomic_thread_fence(memory_order_release);
            return;
        }
    }
}

static void queue_reset(void) {
    pthread_mutex_lock(&q_mu);
    node_t* cur = q_head;
    while (cur) {
        node_t* next = cur->next;
        free(cur);
        cur = next;
    }
    q_head = q_tail = NULL;
    pthread_mutex_unlock(&q_mu);
}

static void queue_push(job_t j) {
    node_t* n = (node_t*)malloc(sizeof(node_t));
    if (!n) {
        perror("malloc");
        return;
    }
    n->job = j;
    n->next = NULL;

    pthread_mutex_lock(&q_mu);
    if (q_tail) {
        q_tail->next = n;
    } else {
        q_head = n;
    }
    q_tail = n;
    pthread_cond_signal(&q_cv);
    pthread_mutex_unlock(&q_mu);

    atomic_fetch_add_explicit(&pending_jobs, 1, memory_order_release);
}

static int queue_pop(job_t* out) {
    pthread_mutex_lock(&q_mu);
    while (!q_head && !atomic_load(&stop_flag)) {
        pthread_cond_wait(&q_cv, &q_mu);
    }
    if (!q_head) {
        pthread_mutex_unlock(&q_mu);
        return 0;
    }
    node_t* n = q_head;
    q_head = n->next;
    if (!q_head) q_tail = NULL;
    *out = n->job;
    free(n);
    pthread_mutex_unlock(&q_mu);
    return 1;
}

// Simulated “analysis”: hash + short sleep to emulate heavy work
static void analyze_and_cache(const char* path, time_t mt) {
    char key[QALLOW_KEY_MAX];
    snprintf(key, sizeof(key), "%s|%ld", path, (long)mt);
    char hit[QALLOW_VAL_MAX];
    if (cache_get(key, hit, sizeof(hit))) {
        qallow_infof("[Qallow] cache hit: %s -> %s\n", path, hit);
        return;
    }
    // heavy work placeholder
    struct timespec ts = {.tv_sec = 0, .tv_nsec = 50 * 1000 * 1000}; // 50 ms
    nanosleep(&ts, NULL);
    char val[QALLOW_VAL_MAX];
    snprintf(val, sizeof(val), "hint:%08lx", (unsigned long)fnv1a64(path, strlen(path)));
    cache_put(key, val);
    qallow_infof("[Qallow] cached: %s -> %s\n", path, val);
}

static void* worker(void* arg) {
    (void)arg;
    job_t j;
    while (queue_pop(&j)) {
        analyze_and_cache(j.path, j.mtime);
        atomic_fetch_sub_explicit(&pending_jobs, 1, memory_order_acq_rel);
    }
    return NULL;
}

static time_t path_mtime(const char* path) {
    struct stat st;
    if (stat(path, &st) == 0) return st.st_mtime;
    return 0;
}

static size_t resolve_thread_count(size_t requested) {
    size_t threads = requested;
    if (threads == 0) {
        threads = (size_t)sysconf(_SC_NPROCESSORS_ONLN);
    }
    if (threads < 2) threads = 2;
    return threads;
}

static void enqueue_files(const phase13_accel_config_t* cfg) {
    if (!cfg || !cfg->files || cfg->file_count == 0) return;
    for (size_t i = 0; i < cfg->file_count; i++) {
        const char* path = cfg->files[i];
        if (!path || !*path) continue;
        job_t j;
        memset(&j, 0, sizeof(j));
        snprintf(j.path, sizeof(j.path), "%s", path);
        j.mtime = path_mtime(j.path);
        queue_push(j);
    }
}

static int accelerator_run(const phase13_accel_config_t* cfg) {
    if (!cfg) {
        errno = EINVAL;
        return -1;
    }

    queue_reset();
    atomic_store(&stop_flag, 0);
    atomic_store(&pending_jobs, 0);
    atomic_store(&remote_sync_seq, 0);

    remote_sync_state_t remote_sync;
    remote_sync_prepare(cfg, &remote_sync);

    g_threads = resolve_thread_count(cfg->thread_count);

    g_cache = cache_attach();
    if (!g_cache) {
        return -1;
    }

    pthread_t* th = (pthread_t*)malloc(sizeof(pthread_t) * g_threads);
    if (!th) {
        perror("malloc");
        return -1;
    }

    size_t launched = 0;
    for (size_t t = 0; t < g_threads; t++) {
        if (pthread_create(&th[t], NULL, worker, NULL) != 0) {
            perror("pthread_create");
            atomic_store(&stop_flag, 1);
            pthread_cond_broadcast(&q_cv);
            for (size_t j = 0; j < launched; j++) pthread_join(th[j], NULL);
            free(th);
            return -1;
        }
        launched++;
    }

    enqueue_files(cfg);
    remote_sync_start(&remote_sync);

    const char* watch_dir = (cfg->watch_dir && cfg->watch_dir[0]) ? cfg->watch_dir : NULL;
    int ifd = -1;
    int wd = -1;
    char* evbuf = NULL;

    if (watch_dir) {
        ifd = inotify_init1(IN_NONBLOCK);
        if (ifd < 0) {
            perror("inotify_init1");
            watch_dir = NULL;
        } else {
            wd = inotify_add_watch(ifd, watch_dir, IN_CLOSE_WRITE | IN_MOVED_TO);
            if (wd < 0) {
                perror("inotify_add_watch");
                close(ifd);
                ifd = -1;
                watch_dir = NULL;
            } else {
                evbuf = (char*)malloc(QALLOW_EVENT_BUFSZ);
                if (!evbuf) {
                    perror("malloc");
                    inotify_rm_watch(ifd, wd);
                    close(ifd);
                    ifd = -1;
                    watch_dir = NULL;
                } else {
                    qallow_infof("[Qallow] watching: %s\n", watch_dir);
                }
            }
        }
    }

    const int keep_running = cfg->keep_running || (watch_dir != NULL);
    const struct timespec idle = {.tv_sec = 0, .tv_nsec = 20 * 1000 * 1000};

    while (keep_running || atomic_load_explicit(&pending_jobs, memory_order_acquire) > 0) {
        if (watch_dir && ifd >= 0) {
            int rd = read(ifd, evbuf, QALLOW_EVENT_BUFSZ);
            if (rd > 0) {
                int off = 0;
                while (off < rd) {
                    struct inotify_event* ev = (struct inotify_event*)(evbuf + off);
                    if (ev->len && (ev->mask & (IN_CLOSE_WRITE | IN_MOVED_TO))) {
                        char path[768];
                        snprintf(path, sizeof(path), "%s/%s", watch_dir, ev->name);
                        job_t j;
                        memset(&j, 0, sizeof(j));
                        snprintf(j.path, sizeof(j.path), "%s", path);
                        j.mtime = path_mtime(j.path);
                        queue_push(j);
                    }
                    off += (int)(sizeof(struct inotify_event) + ev->len);
                }
            }
        }

        if (!keep_running && atomic_load_explicit(&pending_jobs, memory_order_acquire) == 0) {
            break;
        }

        nanosleep(&idle, NULL);
    }

    atomic_store(&stop_flag, 1);
    remote_sync_stop(&remote_sync);
    pthread_cond_broadcast(&q_cv);
    for (size_t t = 0; t < launched; t++) {
        pthread_join(th[t], NULL);
    }
    free(th);

    if (evbuf) free(evbuf);
    if (ifd >= 0) {
        if (wd >= 0) inotify_rm_watch(ifd, wd);
        close(ifd);
    }

    return 0;
}

int qallow_phase13_accel_start(const phase13_accel_config_t* config) {
    return accelerator_run(config);
}

static void usage(const char* argv0) {
    fprintf(stderr,
            "Usage: %s [--threads=N|auto] [--watch=DIR] [--no-watch] [--file=PATH]...\n"
            "           [--remote-sync[=ENDPOINT]] [--remote-sync-interval=SECONDS] [--quiet]\n",
            argv0);
}

int qallow_phase13_main(int argc, char** argv) {
    phase13_accel_config_t cfg = {
        .thread_count = 0,
        .watch_dir = NULL,
        .files = NULL,
        .file_count = 0,
        .keep_running = 0,
        .remote_sync_enabled = 0,
        .remote_sync_endpoint = NULL,
        .remote_sync_interval_sec = 0,
    };

    const char* quiet_env = getenv("QALLOW_QUIET");
    if (quiet_env && quiet_env[0] != '\0' && quiet_env[0] != '0') {
        g_quiet = 1;
    }

    const char* file_args[argc > 1 ? (size_t)argc : 1];
    size_t file_count = 0;

    for (int i = 1; i < argc; i++) {
        const char* arg = argv[i];
        if (strncmp(arg, "--threads=", 10) == 0) {
            const char* value = arg + 10;
            if (strcmp(value, "auto") == 0) {
                cfg.thread_count = 0;
            } else {
                char* end = NULL;
                unsigned long v = strtoul(value, &end, 10);
                if (!end || *end != '\0') {
                    fprintf(stderr, "[ERROR] Invalid --threads value: %s\n", value);
                    return 1;
                }
                cfg.thread_count = (size_t)v;
            }
        } else if (strcmp(arg, "--threads") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "[ERROR] Missing value for --threads\n");
                return 1;
            }
            const char* value = argv[++i];
            if (strcmp(value, "auto") == 0) {
                cfg.thread_count = 0;
            } else {
                char* end = NULL;
                unsigned long v = strtoul(value, &end, 10);
                if (!end || *end != '\0') {
                    fprintf(stderr, "[ERROR] Invalid --threads value: %s\n", value);
                    return 1;
                }
                cfg.thread_count = (size_t)v;
            }
        } else if (strncmp(arg, "--watch=", 8) == 0) {
            cfg.watch_dir = arg + 8;
        } else if (strcmp(arg, "--watch") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "[ERROR] Missing value for --watch\n");
                return 1;
            }
            cfg.watch_dir = argv[++i];
        } else if (strcmp(arg, "--no-watch") == 0) {
            cfg.watch_dir = NULL;
            cfg.keep_running = 0;
        } else if (strncmp(arg, "--file=", 7) == 0) {
            file_args[file_count++] = arg + 7;
        } else if (strcmp(arg, "--file") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "[ERROR] Missing value for --file\n");
                return 1;
            }
            file_args[file_count++] = argv[++i];
        } else if (strncmp(arg, "--remote-sync=", 14) == 0) {
            cfg.remote_sync_enabled = 1;
            cfg.remote_sync_endpoint = arg + 14;
        } else if (strcmp(arg, "--remote-sync") == 0) {
            cfg.remote_sync_enabled = 1;
            if (i + 1 < argc && argv[i + 1] && strncmp(argv[i + 1], "--", 2) != 0) {
                cfg.remote_sync_endpoint = argv[++i];
            }
        } else if (strncmp(arg, "--remote-sync-interval=", 23) == 0) {
            const char* value = arg + 23;
            char* end = NULL;
            unsigned long v = strtoul(value, &end, 10);
            if (!end || *end != '\0' || v == 0) {
                fprintf(stderr, "[ERROR] Invalid --remote-sync-interval value: %s\n", value);
                return 1;
            }
            cfg.remote_sync_interval_sec = (unsigned int)v;
        } else if (strcmp(arg, "--remote-sync-interval") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "[ERROR] Missing value for --remote-sync-interval\n");
                return 1;
            }
            const char* value = argv[++i];
            char* end = NULL;
            unsigned long v = strtoul(value, &end, 10);
            if (!end || *end != '\0' || v == 0) {
                fprintf(stderr, "[ERROR] Invalid --remote-sync-interval value: %s\n", value);
                return 1;
            }
            cfg.remote_sync_interval_sec = (unsigned int)v;
        } else if (strcmp(arg, "--quiet") == 0 || strcmp(arg, "-q") == 0) {
            g_quiet = 1;
        } else if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else if (strncmp(arg, "--", 2) == 0) {
            fprintf(stderr, "[ERROR] Unknown option: %s\n", arg);
            usage(argv[0]);
            return 1;
        } else {
            file_args[file_count++] = arg;
        }
    }

    if (cfg.watch_dir) {
        cfg.keep_running = 1;
    }
    if (cfg.remote_sync_enabled && cfg.keep_running == 0) {
        cfg.keep_running = 1;
    }
    cfg.files = file_args;
    cfg.file_count = file_count;

    return qallow_phase13_accel_start(&cfg);
}

#ifndef QALLOW_PHASE13_EMBEDDED
int main(int argc, char** argv) {
    return qallow_phase13_main(argc, argv);
}
#endif
