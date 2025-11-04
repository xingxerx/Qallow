#include "qallow/module.h"
#include <math.h>
#include <string.h>
#include <stdio.h>


typedef struct {
    double amplitude[4];  // 2-qubit state amplitudes
    double phase[4];      // Phase angles
} quantum_state_t;

static quantum_state_t q_state = {
    .amplitude = {0.5, 0.5, 0.5, 0.5},
    .phase = {0, 0, 0, 0}
};


static void init_quantum_state(ql_state *S) {

    double norm = sqrt(S->energy * S->energy + S->risk * S->risk +
                       S->reward * S->reward + 0.1);

    q_state.amplitude[0] = S->energy / norm;
    q_state.amplitude[1] = S->risk / norm;
    q_state.amplitude[2] = S->reward / norm;
    q_state.amplitude[3] = 0.1 / norm;


    for (int i = 0; i < 4; i++) {
        q_state.phase[i] = S->t * 0.1 * (i + 1);
    }
}


static void hadamard_gate(quantum_state_t *qs) {
    double tmp[4];
    for (int i = 0; i < 4; i++) {
        tmp[i] = qs->amplitude[i];
    }

    for (int i = 0; i < 4; i++) {
        qs->amplitude[i] = 0.0;
        for (int j = 0; j < 4; j++) {
            qs->amplitude[i] += tmp[j] / 2.0;
        }
    }
}


static void phase_gate(quantum_state_t *qs, double angle) {
    for (int i = 0; i < 4; i++) {
        qs->phase[i] += angle;
    }
}


static double measure_quantum(quantum_state_t *qs) {
    double prob_sum = 0.0;
    for (int i = 0; i < 4; i++) {
        prob_sum += qs->amplitude[i] * qs->amplitude[i];
    }


    double result = 0.0;
    for (int i = 0; i < 4; i++) {
        result += (qs->amplitude[i] * qs->amplitude[i] / prob_sum) * (i / 4.0);
    }
    return result;
}


ql_status mod_quantum_predict(ql_state *S) {
    init_quantum_state(S);


    hadamard_gate(&q_state);
    phase_gate(&q_state, S->energy * M_PI);
    hadamard_gate(&q_state);


    double quantum_result = measure_quantum(&q_state);
    S->reward = 0.7 * S->reward + 0.3 * quantum_result;

    return (ql_status){0, "quantum predict ok"};
}


ql_status mod_quantum_optimize(ql_state *S) {
    init_quantum_state(S);


    for (int iter = 0; iter < 3; iter++) {
        hadamard_gate(&q_state);


        for (int i = 0; i < 4; i++) {
            if (q_state.amplitude[i] > 0.4) {
                q_state.phase[i] += M_PI;
            }
        }

        hadamard_gate(&q_state);
    }


    double best_amp = 0.0;
    int best_idx = 0;
    for (int i = 0; i < 4; i++) {
        if (fabs(q_state.amplitude[i]) > best_amp) {
            best_amp = fabs(q_state.amplitude[i]);
            best_idx = i;
        }
    }


    if (best_idx == 0) {
        S->energy += 0.1;
    } else if (best_idx == 1) {
        S->risk -= 0.1;
    } else if (best_idx == 2) {
        S->reward += 0.05;
    }

    return (ql_status){0, "quantum optimize ok"};
}


ql_status mod_hybrid_optimize(ql_state *S) {

    double classical_energy = S->energy;
    double classical_risk = S->risk;


    double grad_energy = (S->reward > 0.5) ? 0.05 : -0.05;
    double grad_risk = (S->reward > 0.5) ? -0.05 : 0.05;

    classical_energy += grad_energy;
    classical_risk += grad_risk;


    init_quantum_state(S);
    hadamard_gate(&q_state);

    double quantum_energy = measure_quantum(&q_state);
    double quantum_risk = measure_quantum(&q_state);


    S->energy = 0.6 * classical_energy + 0.4 * quantum_energy;
    S->risk = 0.6 * classical_risk + 0.4 * quantum_risk;


    S->energy = fmax(0.0, fmin(1.0, S->energy));
    S->risk = fmax(-0.1, fmin(1.0, S->risk));

    return (ql_status){0, "hybrid optimize ok"};
}


ql_status mod_quantum_entangle(ql_state *S) {
    static double entangled_reward = 0.0;
    static double entanglement_strength = 0.0;


    entanglement_strength = fmin(1.0, entanglement_strength + 0.01);


    entangled_reward = (1.0 - entanglement_strength) * S->reward +
                       entanglement_strength * entangled_reward;


    S->reward = entangled_reward;

    return (ql_status){0, "quantum entangle ok"};
}

