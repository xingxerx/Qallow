#include "qallow/module.h"
#include <math.h>

// Multi-objective optimizer balances reward, risk, and energy stability.
ql_status mod_multi_objective_opt(ql_state *S) {
    static double momentum_reward = 0.0;
    static double momentum_risk = 0.0;
    static double momentum_energy = 0.0;

    const double target_reward = 0.35;
    const double target_risk = 0.20;
    const double target_energy = 0.60;
    const double step = 0.045;
    const double momentum = 0.78;

    double reward_grad = target_reward - S->reward;
    double risk_grad = S->risk - target_risk;
    double energy_grad = target_energy - S->energy;

    double coupling = 0.5 * risk_grad - 0.25 * reward_grad;
    double stability_penalty = fabs(S->energy - target_energy) * 0.12;

    momentum_reward = momentum * momentum_reward + (1.0 - momentum) * reward_grad;
    momentum_risk = momentum * momentum_risk + (1.0 - momentum) * (risk_grad + coupling);
    momentum_energy = momentum * momentum_energy + (1.0 - momentum) * (energy_grad - stability_penalty);

    S->reward += step * momentum_reward;
    S->risk -= step * momentum_risk;
    S->energy += step * momentum_energy;

    if (S->reward > 0.9) S->reward = 0.9;
    if (S->reward < -0.5) S->reward = -0.5;
    if (S->risk < 0.0) S->risk = 0.0;
    if (S->risk > 1.0) S->risk = 1.0;
    if (S->energy < 0.0) S->energy = 0.0;
    if (S->energy > 1.0) S->energy = 1.0;

    if (fabs(S->reward - target_reward) < 0.01 && fabs(S->risk - target_risk) < 0.01) {
        S->reward *= 0.98;
    }

    return (ql_status){0, "multi-objective optimizer ok"};
}

// Safety projection nudges state back inside acceptable envelope.
ql_status mod_safety_projection(ql_state *S) {
    const double min_energy = 0.15;
    const double max_energy = 0.85;
    const double max_risk = 0.75;

    if (S->energy < min_energy) {
        S->energy = min_energy + 0.1 * (min_energy - S->energy);
    } else if (S->energy > max_energy) {
        S->energy = max_energy - 0.1 * (S->energy - max_energy);
    }

    if (S->risk > max_risk) {
        S->risk = max_risk - 0.2 * (S->risk - max_risk);
        S->reward -= 0.05;
    }

    if (S->reward < -0.3) {
        S->reward = -0.3;
    }

    return (ql_status){0, "safety projection ok"};
}
