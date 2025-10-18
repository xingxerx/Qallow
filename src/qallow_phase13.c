// src/qallow_phase13.c
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
#include <errno.h>
#include <time.h>

#ifndef QALLOW_CACHE_ENTRIES
#define QALLOW_CACHE_ENTRIES 2048
#endif
#define QALLOW_KEY_MAX 128
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

static cache_t* g_cache = NULL;
static size_t g_threads = 0;

static uint64_t fnv1a64(const void* data, size_t len){
  const uint8_t* p = (const uint8_t*)data;
  uint64_t h = 1469598103934665603ull;
  for(size_t i=0;i<len;i++){ h ^= p[i]; h *= 1099511628211ull; }
  return h;
}

static cache_t* cache_attach(void){
  int fd = shm_open(QALLOW_SHM_NAME, O_CREAT|O_RDWR, 0600);
  if(fd < 0){ perror("shm_open"); exit(1); }
  size_t sz = sizeof(cache_t);
  if(ftruncate(fd, sz) < 0){ perror("ftruncate"); exit(1); }
  void* p = mmap(NULL, sz, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
  if(p == MAP_FAILED){ perror("mmap"); exit(1); }
  close(fd);
  return (cache_t*)p;
}

// linear-probe insert/find
static int cache_get(const char* key, char* out, size_t out_sz){
  uint64_t h = fnv1a64(key, strlen(key));
  size_t idx = h % QALLOW_CACHE_ENTRIES;
  for(size_t i=0;i<QALLOW_CACHE_ENTRIES;i++){
    size_t j = (idx + i) % QALLOW_CACHE_ENTRIES;
    uint64_t tag = atomic_load_explicit(&g_cache->slots[j].tag, memory_order_acquire);
    if(tag == 0) return 0; // stop at empty
    if(tag == h && strncmp(g_cache->slots[j].key, key, QALLOW_KEY_MAX)==0){
      strncpy(out, g_cache->slots[j].val, out_sz-1); out[out_sz-1]=0;
      return 1;
    }
  }
  return 0;
}

static void cache_put(const char* key, const char* val){
  uint64_t h = fnv1a64(key, strlen(key));
  size_t idx = h % QALLOW_CACHE_ENTRIES;
  for(size_t i=0;i<QALLOW_CACHE_ENTRIES;i++){
    size_t j = (idx + i) % QALLOW_CACHE_ENTRIES;
    uint64_t expect = 0;
    if(atomic_compare_exchange_strong(&g_cache->slots[j].tag, &expect, h)){
      strncpy(g_cache->slots[j].key, key, QALLOW_KEY_MAX-1);
      g_cache->slots[j].key[QALLOW_KEY_MAX-1]=0;
      strncpy(g_cache->slots[j].val, val, QALLOW_VAL_MAX-1);
      g_cache->slots[j].val[QALLOW_VAL_MAX-1]=0;
      atomic_thread_fence(memory_order_release);
      return;
    }
    if(expect == h && strncmp(g_cache->slots[j].key, key, QALLOW_KEY_MAX)==0){
      strncpy(g_cache->slots[j].val, val, QALLOW_VAL_MAX-1);
      g_cache->slots[j].val[QALLOW_VAL_MAX-1]=0;
      atomic_thread_fence(memory_order_release);
      return;
    }
  }
}

typedef struct job_s { char path[512]; time_t mtime; } job_t;

typedef struct node_s {
  job_t job; struct node_s* next;
} node_t;

static node_t* q_head = NULL;
static node_t* q_tail = NULL;
static pthread_mutex_t q_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  q_cv = PTHREAD_COND_INITIALIZER;
static atomic_int stop_flag = 0;

static void queue_push(job_t j){
  node_t* n = (node_t*)malloc(sizeof(node_t));
  n->job = j; n->next = NULL;
  pthread_mutex_lock(&q_mu);
  if(q_tail) q_tail->next = n; else q_head = n;
  q_tail = n;
  pthread_cond_signal(&q_cv);
  pthread_mutex_unlock(&q_mu);
}

static int queue_pop(job_t* out){
  pthread_mutex_lock(&q_mu);
  while(!q_head && !atomic_load(&stop_flag)) pthread_cond_wait(&q_cv, &q_mu);
  if(!q_head){ pthread_mutex_unlock(&q_mu); return 0; }
  node_t* n = q_head; q_head = n->next; if(!q_head) q_tail = NULL;
  *out = n->job; free(n);
  pthread_mutex_unlock(&q_mu);
  return 1;
}

// Simulated “analysis”: hash + short sleep to emulate heavy work
static void analyze_and_cache(const char* path, time_t mt){
  char key[QALLOW_KEY_MAX]; snprintf(key, sizeof(key), "%s|%ld", path, (long)mt);
  char hit[QALLOW_VAL_MAX];
  if(cache_get(key, hit, sizeof(hit))){
    printf("[Qallow] cache hit: %s -> %s\n", path, hit);
    return;
  }
  // heavy work placeholder
  struct timespec ts = {.tv_sec=0, .tv_nsec=50*1000*1000}; // 50 ms
  nanosleep(&ts, NULL);
  char val[QALLOW_VAL_MAX];
  snprintf(val, sizeof(val), "hint:%08lx", (unsigned long)fnv1a64(path, strlen(path)));
  cache_put(key, val);
  printf("[Qallow] cached: %s -> %s\n", path, val);
}

static void* worker(void* arg){
  (void)arg;
  job_t j;
  while(queue_pop(&j)){
    analyze_and_cache(j.path, j.mtime);
  }
  return NULL;
}

static time_t path_mtime(const char* path){
  struct stat st;
  if(stat(path, &st)==0) return st.st_mtime;
  return 0;
}

static void usage(const char* argv0){
  fprintf(stderr, "Usage: %s [--threads=N] [--watch=DIR] [--file=PATH]*\n", argv0);
}

int main(int argc, char** argv){
  const char* watch_dir = NULL;
  g_threads = sysconf(_SC_NPROCESSORS_ONLN);
  if(g_threads < 2) g_threads = 2;

  for(int i=1;i<argc;i++){
    if(strncmp(argv[i], "--threads=",10)==0) g_threads = strtoul(argv[i]+10,NULL,10);
    else if(strncmp(argv[i], "--watch=",8)==0) watch_dir = argv[i]+8;
    else if(strncmp(argv[i], "--file=",7)==0){
      job_t j; memset(&j,0,sizeof(j));
      strncpy(j.path, argv[i]+7, sizeof(j.path)-1);
      j.mtime = path_mtime(j.path);
      queue_push(j);
    } else { usage(argv[0]); }
  }

  g_cache = cache_attach();

  // start workers
  pthread_t* th = (pthread_t*)malloc(sizeof(pthread_t)*g_threads);
  for(size_t t=0;t<g_threads;t++) pthread_create(&th[t], NULL, worker, NULL);

  // optional watch mode
  int ifd = -1, wd = -1;
  char* evbuf = NULL;
  if(watch_dir){
    ifd = inotify_init1(IN_NONBLOCK);
    if(ifd < 0){ perror("inotify_init1"); }
    else {
      wd = inotify_add_watch(ifd, watch_dir, IN_CLOSE_WRITE | IN_MOVED_TO);
      if(wd < 0) perror("inotify_add_watch");
      evbuf = (char*)malloc(QALLOW_EVENT_BUFSZ);
      printf("[Qallow] watching: %s\n", watch_dir);
    }
  }

  // event loop
  while(1){
    if(ifd >= 0){
      int rd = read(ifd, evbuf, QALLOW_EVENT_BUFSZ);
      if(rd > 0){
        int off = 0;
        while(off < rd){
          struct inotify_event* ev = (struct inotify_event*)(evbuf + off);
          if(ev->len && (ev->mask & (IN_CLOSE_WRITE|IN_MOVED_TO))){
            char path[768];
            snprintf(path, sizeof(path), "%s/%s", watch_dir, ev->name);
            job_t j; memset(&j,0,sizeof(j));
            strncpy(j.path, path, sizeof(j.path)-1);
            j.mtime = path_mtime(j.path);
            queue_push(j);
          }
          off += sizeof(struct inotify_event) + ev->len;
        }
      }
    }
    // small idle
    struct timespec ts = {.tv_sec=0, .tv_nsec=20*1000*1000};
    nanosleep(&ts, NULL);
  }

  // never reached in typical service mode
  atomic_store(&stop_flag, 1);
  pthread_cond_broadcast(&q_cv);
  for(size_t t=0;t<g_threads;t++) pthread_join(th[t], NULL);
  free(th);
  if(evbuf) free(evbuf);
  if(ifd>=0){ if(wd>=0) inotify_rm_watch(ifd, wd); close(ifd); }
  return 0;
}
