#include "qallow/performance_profiler.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>

/* Get current timestamp in microseconds */
static int64_t qallow_get_usec_timestamp(void) {
#ifdef _WIN32
    /* Windows implementation */
    return 0;  /* TODO: Implement Windows high-resolution timer */
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (int64_t)tv.tv_sec * 1000000LL + (int64_t)tv.tv_usec;
#endif
}

qallow_profiler_t qallow_profiler_init(void) {
    qallow_profiler_t prof;
    memset(&prof, 0, sizeof(prof));
    prof.start_clock = clock();
    prof.start_usec = qallow_get_usec_timestamp();
    return prof;
}

void qallow_profiler_mark(qallow_profiler_t* prof, const char* label) {
    if (!prof || !label || prof->mark_count >= 256) {
        return;
    }

    int idx = prof->mark_count++;
    strncpy(prof->marks[idx].label, label, sizeof(prof->marks[idx].label) - 1);
    prof->marks[idx].label[sizeof(prof->marks[idx].label) - 1] = '\0';
    prof->marks[idx].clock_time = clock();
    prof->marks[idx].usec_timestamp = qallow_get_usec_timestamp();

    /* Calculate elapsed time since start */
    int64_t elapsed_usec = prof->marks[idx].usec_timestamp - prof->start_usec;
    prof->marks[idx].elapsed_ms = elapsed_usec / 1000.0;
}

void qallow_profiler_record_phase(qallow_profiler_t* prof,
                                   int phase_num,
                                   double elapsed_ms) {
    if (!prof) return;

    switch (phase_num) {
        case 12:
            prof->phase12_time_ms = elapsed_ms;
            break;
        case 13:
            prof->phase13_time_ms = elapsed_ms;
            break;
        case 14:
            prof->phase14_time_ms = elapsed_ms;
            break;
        case 15:
            prof->phase15_time_ms = elapsed_ms;
            break;
        default:
            break;
    }
}

void qallow_profiler_record_gpu_cpu_ratio(qallow_profiler_t* prof,
                                          double gpu_ms,
                                          double cpu_ms) {
    if (!prof || cpu_ms <= 0.0) return;
    prof->gpu_vs_cpu_ratio = gpu_ms / cpu_ms;
}

int qallow_profiler_export_csv(const qallow_profiler_t* prof,
                                const char* csv_path) {
    if (!prof || !csv_path) {
        return -1;
    }

    FILE* f = fopen(csv_path, "w");
    if (!f) {
        return -1;
    }

    /* Write header */
    fprintf(f, "mark_index,label,elapsed_ms,phase_summary\n");

    /* Write marks */
    for (int i = 0; i < prof->mark_count; ++i) {
        fprintf(f, "%d,\"%s\",%.3f,\n",
                i, prof->marks[i].label, prof->marks[i].elapsed_ms);
    }

    /* Write phase summary */
    fprintf(f, "\n# Phase Timing Summary\n");
    fprintf(f, "phase,time_ms\n");
    if (prof->phase12_time_ms > 0.0) {
        fprintf(f, "12,%.3f\n", prof->phase12_time_ms);
    }
    if (prof->phase13_time_ms > 0.0) {
        fprintf(f, "13,%.3f\n", prof->phase13_time_ms);
    }
    if (prof->phase14_time_ms > 0.0) {
        fprintf(f, "14,%.3f\n", prof->phase14_time_ms);
    }
    if (prof->phase15_time_ms > 0.0) {
        fprintf(f, "15,%.3f\n", prof->phase15_time_ms);
    }

    /* GPU vs CPU ratio */
    if (prof->gpu_vs_cpu_ratio > 0.0) {
        fprintf(f, "\n# GPU/CPU Performance\n");
        fprintf(f, "gpu_vs_cpu_ratio,%.2f\n", prof->gpu_vs_cpu_ratio);
    }

    fclose(f);
    return 0;
}

int qallow_profiler_export_json(const qallow_profiler_t* prof,
                                 const char* json_path) {
    if (!prof || !json_path) {
        return -1;
    }

    FILE* f = fopen(json_path, "w");
    if (!f) {
        return -1;
    }

    /* Write JSON header */
    fprintf(f, "{\n");
    fprintf(f, "  \"marks\": [\n");

    /* Write marks */
    for (int i = 0; i < prof->mark_count; ++i) {
        fprintf(f, "    {\"index\": %d, \"label\": \"%s\", \"elapsed_ms\": %.3f}",
                i, prof->marks[i].label, prof->marks[i].elapsed_ms);
        if (i < prof->mark_count - 1) {
            fprintf(f, ",");
        }
        fprintf(f, "\n");
    }

    fprintf(f, "  ],\n");
    fprintf(f, "  \"phases\": {\n");
    fprintf(f, "    \"phase12_ms\": %.3f,\n", prof->phase12_time_ms);
    fprintf(f, "    \"phase13_ms\": %.3f,\n", prof->phase13_time_ms);
    fprintf(f, "    \"phase14_ms\": %.3f,\n", prof->phase14_time_ms);
    fprintf(f, "    \"phase15_ms\": %.3f\n", prof->phase15_time_ms);
    fprintf(f, "  },\n");
    fprintf(f, "  \"gpu_vs_cpu_ratio\": %.2f\n", prof->gpu_vs_cpu_ratio);
    fprintf(f, "}\n");

    fclose(f);
    return 0;
}

void qallow_profiler_print_summary(const qallow_profiler_t* prof) {
    if (!prof) return;

    printf("\n=== Performance Profile Summary ===\n");
    printf("Phase Execution Times:\n");
    if (prof->phase12_time_ms > 0.0) {
        printf("  Phase 12: %.3f ms\n", prof->phase12_time_ms);
    }
    if (prof->phase13_time_ms > 0.0) {
        printf("  Phase 13: %.3f ms\n", prof->phase13_time_ms);
    }
    if (prof->phase14_time_ms > 0.0) {
        printf("  Phase 14: %.3f ms\n", prof->phase14_time_ms);
    }
    if (prof->phase15_time_ms > 0.0) {
        printf("  Phase 15: %.3f ms\n", prof->phase15_time_ms);
    }

    if (prof->gpu_vs_cpu_ratio > 0.0) {
        printf("\nGPU vs CPU Performance:\n");
        printf("  Ratio: %.2fx\n", prof->gpu_vs_cpu_ratio);
        if (prof->gpu_vs_cpu_ratio < 1.0) {
            printf("  Status: GPU is %.0f%% faster\n",
                   (1.0 - prof->gpu_vs_cpu_ratio) * 100.0);
        } else {
            printf("  Status: CPU is %.0f%% faster\n",
                   (prof->gpu_vs_cpu_ratio - 1.0) * 100.0);
        }
    }

    printf("\nTotal Marks Recorded: %d\n", prof->mark_count);
    printf("===================================\n\n");
}

double qallow_profiler_elapsed_ms(const qallow_profiler_t* prof) {
    if (!prof) return 0.0;
    int64_t now = qallow_get_usec_timestamp();
    int64_t elapsed_usec = now - prof->start_usec;
    return elapsed_usec / 1000.0;
}
