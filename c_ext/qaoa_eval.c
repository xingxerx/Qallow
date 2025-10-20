#include <math.h>
#include <stddef.h>
#include <stdint.h>

double qallow_qaoa_eval_score(const char* bitstring, double probability, int epochs) {
    if (!bitstring || probability < 0.0) {
        return 0.0;
    }

    double score = probability;
    for (const char* c = bitstring; *c; ++c) {
        if (*c == '1') {
            score += 0.05;
        } else if (*c == '0') {
            score += 0.02;
        }
    }

    score *= (1.0 + log1p((double)epochs));
    return score;
}
