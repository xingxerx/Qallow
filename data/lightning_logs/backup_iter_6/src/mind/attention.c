#include "qallow/module.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

// Attention head: computes weighted importance of each module output
typedef struct {
    double query[8];      // Query vector (module importance)
    double key[8];        // Key vector (module signature)
    double value[8];      // Value vector (module output)
    double attention[8];  // Attention weights
} attention_head_t;

static attention_head_t heads[4] = {0};

// Softmax for attention weights
static void softmax(double *x, int n) {
    double max_x = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_x) max_x = x[i];
    }
    
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        x[i] = exp(x[i] - max_x);
        sum += x[i];
    }
    
    for (int i = 0; i < n; i++) {
        x[i] /= sum;
    }
}

// Compute attention scores (query · key)
static void compute_attention(attention_head_t *head, int n) {
    for (int i = 0; i < n; i++) {
        head->attention[i] = 0.0;
        for (int j = 0; j < n; j++) {
            head->attention[i] += head->query[j] * head->key[i];
        }
    }
    softmax(head->attention, n);
}

// Multi-head attention module
ql_status mod_attention(ql_state *S) {
    // Initialize attention heads based on state
    for (int h = 0; h < 4; h++) {
        // Query: what are we looking for?
        heads[h].query[0] = S->energy;      // Energy importance
        heads[h].query[1] = 1.0 - S->risk;  // Risk aversion
        heads[h].query[2] = S->reward;      // Reward seeking
        heads[h].query[3] = S->t / 100.0;   // Time awareness
        
        // Key: module signatures
        heads[h].key[0] = 0.9;  // Model signature
        heads[h].key[1] = 0.8;  // Predict signature
        heads[h].key[2] = 0.7;  // Plan signature
        heads[h].key[3] = 0.6;  // Learn signature
        
        compute_attention(&heads[h], 4);
    }
    
    // Aggregate attention across heads
    double avg_attention[4] = {0};
    for (int h = 0; h < 4; h++) {
        for (int i = 0; i < 4; i++) {
            avg_attention[i] += heads[h].attention[i];
        }
    }
    for (int i = 0; i < 4; i++) {
        avg_attention[i] /= 4.0;
    }
    
    // Apply attention-weighted modulation to state
    S->reward *= (0.5 + 0.5 * avg_attention[2]);  // Reward attention
    S->energy *= (0.5 + 0.5 * avg_attention[0]);  // Energy attention
    S->risk *= (0.5 + 0.5 * avg_attention[1]);    // Risk attention
    
    return (ql_status){0, "attention ok"};
}

// Cross-attention: compare current state with historical patterns
ql_status mod_cross_attention(ql_state *S) {
    static double history[100][3] = {0};  // Store last 100 states
    static int history_idx = 0;
    
    // Store current state
    history[history_idx][0] = S->energy;
    history[history_idx][1] = S->risk;
    history[history_idx][2] = S->reward;
    history_idx = (history_idx + 1) % 100;
    
    // Find most similar historical state
    double best_sim = -1.0;
    int best_idx = 0;
    for (int i = 0; i < 100; i++) {
        double sim = 0.0;
        sim += fabs(history[i][0] - S->energy);
        sim += fabs(history[i][1] - S->risk);
        sim += fabs(history[i][2] - S->reward);
        sim = 1.0 / (1.0 + sim);  // Similarity score
        
        if (sim > best_sim) {
            best_sim = sim;
            best_idx = i;
        }
    }
    
    // Blend with historical pattern
    S->energy = 0.8 * S->energy + 0.2 * history[best_idx][0];
    S->risk = 0.8 * S->risk + 0.2 * history[best_idx][1];
    S->reward = 0.8 * S->reward + 0.2 * history[best_idx][2];
    
    return (ql_status){0, "cross-attention ok"};
}

