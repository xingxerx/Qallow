#include "qallow/module.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


typedef struct {
    double local_reward;
    double local_energy;
    double local_risk;
    double global_reward;
    double global_energy;
    double global_risk;
    int node_id;
    int total_nodes;
    int iteration;
} federated_state_t;

static federated_state_t fed_state = {
    .local_reward = 0.0,
    .local_energy = 0.5,
    .local_risk = 0.5,
    .global_reward = 0.0,
    .global_energy = 0.5,
    .global_risk = 0.5,
    .node_id = 0,
    .total_nodes = 3,
    .iteration = 0
};


static void local_train_step(federated_state_t *fs, ql_state *S) {

    double local_grad = (S->reward > fs->local_reward) ? 0.01 : -0.01;

    fs->local_reward += local_grad;
    fs->local_energy = S->energy;
    fs->local_risk = S->risk;


    fs->local_reward = fmax(-1.0, fmin(1.0, fs->local_reward));
}


static void aggregate_models(federated_state_t *fs) {

    fs->global_reward = (fs->global_reward * (fs->total_nodes - 1) + fs->local_reward) / fs->total_nodes;
    fs->global_energy = (fs->global_energy * (fs->total_nodes - 1) + fs->local_energy) / fs->total_nodes;
    fs->global_risk = (fs->global_risk * (fs->total_nodes - 1) + fs->local_risk) / fs->total_nodes;
}


static void broadcast_global_model(federated_state_t *fs, ql_state *S) {

    S->reward = 0.7 * S->reward + 0.3 * fs->global_reward;
    S->energy = 0.7 * S->energy + 0.3 * fs->global_energy;
    S->risk = 0.7 * S->risk + 0.3 * fs->global_risk;
}


ql_status mod_federated_learn(ql_state *S) {

    local_train_step(&fed_state, S);


    if (fed_state.iteration % 10 == 0) {
        aggregate_models(&fed_state);
        broadcast_global_model(&fed_state, S);
    }

    fed_state.iteration++;

    return (ql_status){0, "federated learn ok"};
}


ql_status mod_privacy_preserving_learn(ql_state *S) {

    double noise_scale = 0.01;
    double noise = (rand() % 100 - 50) * noise_scale / 50.0;

    S->reward += noise;
    S->energy += noise * 0.1;


    S->reward = fmax(-1.0, fmin(1.0, S->reward));
    S->energy = fmax(0.0, fmin(1.0, S->energy));

    return (ql_status){0, "privacy learn ok"};
}


ql_status mod_gradient_compression(ql_state *S) {

    int quantization_bits = 8;
    double scale = (1 << quantization_bits) - 1;


    int q_reward = (int)(S->reward * scale);
    S->reward = (double)q_reward / scale;


    int q_energy = (int)(S->energy * scale);
    S->energy = (double)q_energy / scale;

    return (ql_status){0, "gradient compression ok"};
}


ql_status mod_async_param_server(ql_state *S) {
    static double param_buffer[3] = {0.0, 0.5, 0.5};
    static int update_count = 0;


    param_buffer[0] = 0.9 * param_buffer[0] + 0.1 * S->reward;
    param_buffer[1] = 0.9 * param_buffer[1] + 0.1 * S->energy;
    param_buffer[2] = 0.9 * param_buffer[2] + 0.1 * S->risk;


    S->reward = param_buffer[0];
    S->energy = param_buffer[1];
    S->risk = param_buffer[2];

    update_count++;

    return (ql_status){0, "async param server ok"};
}


ql_status mod_consensus(ql_state *S) {
    static double consensus_reward = 0.0;
    static double consensus_energy = 0.5;
    static double consensus_risk = 0.5;


    double avg_reward = (S->reward + consensus_reward) / 2.0;
    double avg_energy = (S->energy + consensus_energy) / 2.0;
    double avg_risk = (S->risk + consensus_risk) / 2.0;


    consensus_reward = avg_reward;
    consensus_energy = avg_energy;
    consensus_risk = avg_risk;


    S->reward = consensus_reward;
    S->energy = consensus_energy;
    S->risk = consensus_risk;

    return (ql_status){0, "consensus ok"};
}

