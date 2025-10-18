#include "ethics_core.h"

#include <math.h>

double ethics_bayes_trust_update(double prior_belief,
                                 double signal_strength,
                                 double beta) {
    double prior = prior_belief;
    if (!isfinite(prior) || prior < 0.0) prior = 0.0;
    if (prior > 1.0) prior = 1.0;

    double signal = signal_strength;
    if (!isfinite(signal) || signal < 0.0) signal = 0.0;
    if (signal > 1.0) signal = 1.0;

    double weight = beta;
    if (!isfinite(weight) || weight < 0.01) weight = 0.01;

    double posterior = (prior * weight + signal) / (weight + 1.0);
    if (posterior < 0.0) posterior = 0.0;
    if (posterior > 1.0) posterior = 1.0;
    return posterior;
}

