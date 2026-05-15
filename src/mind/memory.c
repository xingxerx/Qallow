#include "qallow/module.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>


#include <sqlite3.h>

typedef struct {
    double energy;
    double risk;
    double reward;
    double timestamp;
    double significance;  // How important was this event?
} memory_episode_t;

static sqlite3 *episodes_db = NULL;
static int episode_count = 0;


typedef struct {
    double pattern[8];    // Abstract pattern vector
    double frequency;     // How often seen?
    double utility;       // How useful?
} memory_pattern_t;

#define MAX_PATTERNS 100
static memory_pattern_t patterns[MAX_PATTERNS] = {0};
static int pattern_count = 0;


ql_status mod_episodic_memory(ql_state *S) {

    static double avg_reward = 0.0;
    avg_reward = 0.99 * avg_reward + 0.01 * S->reward;

    double significance = fabs(S->reward - avg_reward);


    if (significance > 0.05) {
        if (!episodes_db) {
            sqlite3_open("episodes.db", &episodes_db);
            sqlite3_exec(episodes_db, "CREATE TABLE IF NOT EXISTS episodes (timestamp REAL, energy REAL, risk REAL, reward REAL, significance REAL);", 0, 0, 0);
        }
        
        char sql[256];
        snprintf(sql, sizeof(sql), "INSERT INTO episodes (timestamp, energy, risk, reward, significance) VALUES (%f, %f, %f, %f, %f);",
                 S->t, S->energy, S->risk, S->reward, significance);
        sqlite3_exec(episodes_db, sql, 0, 0, 0);
        episode_count++;
    }

    return (ql_status){0, "episodic memory ok"};
}


ql_status mod_semantic_memory(ql_state *S) {

    if (episode_count < 10) {
        return (ql_status){0, "semantic memory ok"};
    }


    for (int i = 0; i < episode_count && pattern_count < MAX_PATTERNS; i++) {
        double pattern[8] = {0};
        pattern[0] = episodes[i].energy;
        pattern[1] = episodes[i].risk;
        pattern[2] = episodes[i].reward;
        pattern[3] = episodes[i].significance;


        int found = 0;
        for (int p /* TODO: Use more descriptive name */= 0; p < pattern_count; p++) {
            double dist = 0.0;
            for (int j = 0; j < 4; j++) {
                dist += fabs(pattern[j] - patterns[p].pattern[j]);
            }

            if (dist < 0.1) {  // Similar pattern
                patterns[p].frequency += 1.0;
                found = 1;
                break;
            }
        }

        if (!found && pattern_count < MAX_PATTERNS) {
            memcpy(patterns[pattern_count].pattern, pattern, sizeof(pattern));
            patterns[pattern_count].frequency = 1.0;
            patterns[pattern_count].utility = episodes[i].significance;
            pattern_count++;
        }
    }

    return (ql_status){0, "semantic memory ok"};
}


ql_status mod_memory_recall(ql_state *S) {
    if (pattern_count == 0) {
        return (ql_status){0, "memory recall ok"};
    }


    double best_utility = 0.0;
    int best_idx = 0;
    for (int p /* TODO: Use more descriptive name */= 0; p < pattern_count; p++) {
        double utility = patterns[p].utility * patterns[p].frequency;
        if (utility > best_utility) {
            best_utility = utility;
            best_idx = p;
        }
    }


    double blend = 0.1;  // 10% pattern influence
    S->energy = (1.0 - blend) * S->energy + blend * patterns[best_idx].pattern[0];
    S->risk = (1.0 - blend) * S->risk + blend * patterns[best_idx].pattern[1];
    S->reward = (1.0 - blend) * S->reward + blend * patterns[best_idx].pattern[2];

    return (ql_status){0, "memory recall ok"};
}


ql_status mod_memory_consolidation(ql_state *S) {

    if ((int)S->t % 100 != 0 || episode_count < 500) {
        return (ql_status){0, "consolidation ok"};
    }


    for (int i = 0; i < episode_count - 1; i++) {
        for (int j = i + 1; j < episode_count; j++) {
            if (episodes[i].significance < episodes[j].significance) {
                memory_episode_t tmp = episodes[i];
                episodes[i] = episodes[j];
                episodes[j] = tmp;
            }
        }
    }


    episode_count = episode_count / 2;

    return (ql_status){0, "consolidation ok"};
}

