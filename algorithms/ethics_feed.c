/* Multi-block comment removed */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ethics_core.h"

#define MAX_LINE 512

/* Multi-block comment removed */
int ethics_ingest_signal(const char *path, ethics_metrics_t *metrics) {
    if (!path || !metrics) {
        fprintf(stderr, "[ethics_feed] ERROR: NULL parameters\n");
        return 0;
    }

    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "[ethics_feed] WARNING: Cannot open %s\n", path);
        return 0;
    }


    double vals[10];


    char line[512];
    if (!fgets(line, sizeof(line), f)) {
        fclose(f);
        return 0;
    }

    if (line[0] == '#') {

        if (!fgets(line, sizeof(line), f)) {
            fclose(f);
            return 0;
        }
    }


    int count = sscanf(line, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                      &vals[0], &vals[1], &vals[2], &vals[3], &vals[4],
                      &vals[5], &vals[6], &vals[7], &vals[8], &vals[9]);

    fclose(f);

    if (count != 10) {
        fprintf(stderr, "[ethics_feed] ERROR: Expected 10 values, got %d\n", count);
        return 0;
    }


    metrics->safety = (vals[0] + vals[1] + vals[2]) / 3.0;
    metrics->clarity = (vals[3] + vals[4] + vals[5] + vals[6]) / 4.0;
    metrics->human = (vals[7] + vals[8] + vals[9]) / 3.0;


    double *metric_ptrs[] = {&metrics->safety, &metrics->clarity, &metrics->human};
    const char *names[] = {"safety", "clarity", "human"};

    for (int i = 0; i < 3; i++) {
        if (*metric_ptrs[i] < 0.0 || *metric_ptrs[i] > 1.0) {
            fprintf(stderr, "[ethics_feed] WARNING: %s value out of range [0,1]: %.3f\n",
                    names[i], *metric_ptrs[i]);
            *metric_ptrs[i] = (*metric_ptrs[i] < 0.0) ? 0.0 : 1.0;  // Clamp
        }
    }

    time_t now = time(NULL);
    fprintf(stdout, "[ethics_feed] Ingested at %s", ctime(&now));
    fprintf(stdout, "  Safety:  %.3f (avg of %.3f, %.3f, %.3f)\n",
            metrics->safety, vals[0], vals[1], vals[2]);
    fprintf(stdout, "  Clarity: %.3f (avg of %.3f, %.3f, %.3f, %.3f)\n",
            metrics->clarity, vals[3], vals[4], vals[5], vals[6]);
    fprintf(stdout, "  Human:   %.3f (avg of %.3f, %.3f, %.3f)\n",
            metrics->human, vals[7], vals[8], vals[9]);

    double spread_sc = fabs(metrics->safety - metrics->clarity);
    double spread_ch = fabs(metrics->clarity - metrics->human);
    double spread_sh = fabs(metrics->safety - metrics->human);
    double drift = 0.4 * spread_sc + 0.4 * spread_ch + 0.2 * spread_sh;
    if (drift < 0.0) {
        drift = 0.0;
    } else if (drift > 1.0) {
        drift = 1.0;
    }
    metrics->reality_drift = drift;

    fprintf(stdout, "  Reality Drift: %.3f (spread SC=%.3f CH=%.3f SH=%.3f)\n",
            metrics->reality_drift, spread_sc, spread_ch, spread_sh);

    return 1;
}


void ethics_log_decision(const char *log_path, double score, const char *action) {
    FILE *f = fopen(log_path, "a");
    if (!f) return;

    time_t now = time(NULL);
    fprintf(f, "%ld,%.4f,%s\n", now, score, action ? action : "none");
    fclose(f);
}


int ethics_verify_freshness(const char *path, int max_age_sec) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;


    char line[MAX_LINE];
    if (fgets(line, sizeof(line), f)) {
        if (line[0] == '#') {
            long ts;
            if (sscanf(line, "# %ld", &ts) == 1) {
                time_t now = time(NULL);
                int age = (int)(now - ts);
                fclose(f);
                return (age <= max_age_sec);
            }
        }
    }

    fclose(f);
    return 1;  // No timestamp = assume fresh
}
