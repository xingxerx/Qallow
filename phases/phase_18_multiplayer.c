/* Multi-block comment removed */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define MAX_NODES 16
#define MAX_PEERS 15
#define STATE_VECTOR_DIM 128
#define CONSENSUS_THRESHOLD 0.7

typedef struct {
    int node_id;
    float state_vector[STATE_VECTOR_DIM];
    float confidence;
    int timestamp;
} NodeState;

typedef struct {
    NodeState nodes[MAX_NODES];
    int num_nodes;
    float consensus_vector[STATE_VECTOR_DIM];
    float consensus_strength;
    float synchronization_quality;
    int ledger_entries;
} MultiplayerState;

/* Multi-block comment removed */
void compute_consensus(MultiplayerState *state) {
    // Initialize consensus vector
    for (int i = 0; i < STATE_VECTOR_DIM; i++) {
        state->consensus_vector[i] = 0.0;
    }
    
    // Average all node states
    for (int i = 0; i < state->num_nodes; i++) {
        for (int j = 0; j < STATE_VECTOR_DIM; j++) {
            state->consensus_vector[j] += state->nodes[i].state_vector[j];
        }
    }
    
    for (int i = 0; i < STATE_VECTOR_DIM; i++) {
        state->consensus_vector[i] /= state->num_nodes;
    }
}


float calculate_consensus_strength(MultiplayerState *state) {
    float total_deviation = 0.0;
    
    for (int i = 0; i < state->num_nodes; i++) {
        float deviation = 0.0;
        for (int j = 0; j < STATE_VECTOR_DIM; j++) {
            float diff = state->nodes[i].state_vector[j] - state->consensus_vector[j];
            deviation += diff * diff;
        }
        total_deviation += sqrt(deviation / STATE_VECTOR_DIM);
    }
    
    float avg_deviation = total_deviation / state->num_nodes;
    float strength = 1.0 / (1.0 + avg_deviation);
    
    return strength;
}


int validate_peer_state(NodeState *peer, MultiplayerState *state) {
    // Check if peer state is within acceptable range
    for (int i = 0; i < STATE_VECTOR_DIM; i++) {
        if (peer->state_vector[i] < 0.0 || peer->state_vector[i] > 1.0) {
            return 0;
        }
    }
    
    // Check confidence level
    if (peer->confidence < 0.5) {
        return 0;
    }
    
    return 1;
}


void merge_into_ledger(MultiplayerState *state) {
    state->ledger_entries = state->num_nodes;
    
    // Each node contributes to the ledger
    for (int i = 0; i < state->num_nodes; i++) {
        state->nodes[i].confidence = state->consensus_strength;
    }
}


int main(int argc, char *argv[]) {
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║  Phase 18: Multiplayer Synchronization - Consensus Ledger    ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");

    srand(time(NULL));
    
    MultiplayerState state = {0};
    
    // Initialize nodes
    printf("🌐 Initializing multiplayer nodes...\n");
    int num_nodes = 3 + (rand() % 8);
    state.num_nodes = num_nodes;
    
    for (int i = 0; i < num_nodes; i++) {
        state.nodes[i].node_id = i;
        state.nodes[i].timestamp = time(NULL);
        state.nodes[i].confidence = 0.7 + (rand() / (float)RAND_MAX) * 0.3;
        
        // Initialize state vector
        for (int j = 0; j < STATE_VECTOR_DIM; j++) {
            state.nodes[i].state_vector[j] = (rand() / (float)RAND_MAX);
        }
    }
    
    // Validate peer states
    printf("✅ Validating peer states...\n");
    int valid_peers = 0;
    for (int i = 0; i < num_nodes; i++) {
        if (validate_peer_state(&state.nodes[i], &state)) {
            valid_peers++;
        }
    }
    
    // Compute consensus
    printf("🤝 Computing consensus from %d nodes...\n", valid_peers);
    compute_consensus(&state);
    
    // Calculate consensus strength
    state.consensus_strength = calculate_consensus_strength(&state);
    
    // Check if consensus meets threshold
    if (state.consensus_strength >= CONSENSUS_THRESHOLD) {
        printf("✅ Consensus achieved (strength: %.4f)\n", state.consensus_strength);
    } else {
        printf("⚠️  Weak consensus (strength: %.4f)\n", state.consensus_strength);
    }
    
    // Merge into ledger
    printf("📝 Merging states into shared ledger...\n");
    merge_into_ledger(&state);
    
    // Calculate synchronization quality
    state.synchronization_quality = (valid_peers / (float)num_nodes) * state.consensus_strength;
    
    // Report results
    printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("📈 Multiplayer Synchronization Results:\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("  Total Nodes: %d\n", num_nodes);
    printf("  Valid Peers: %d\n", valid_peers);
    printf("  Consensus Strength: %.4f\n", state.consensus_strength);
    printf("  Synchronization Quality: %.4f\n", state.synchronization_quality);
    printf("  Ledger Entries: %d\n", state.ledger_entries);
    printf("  Status: %s\n", state.synchronization_quality > 0.7 ? "✅ SYNCED" : "⚠️  DIVERGING");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");
    
    printf("✅ Phase 18 Complete: Multiplayer synchronization finished\n");
    return 0;
}

