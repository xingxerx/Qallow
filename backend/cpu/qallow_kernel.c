#include "qallow_kernel.h"
#include "qallow_metrics.h"
#include "ethics.h"
#include "phase14.h"
#include "phase15.h"
#include "overlay.h"
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#ifndef _WIN32
#include <unistd.h>
#include <sys/wait.h>
#endif
#include <errno.h>

#if CUDA_ENABLED
extern void runPhotonicSimulation(double* hostData, int n, unsigned long seed);
extern void runQuantumOptimizer(double* hostData, int n);
#endif

#define QALLOW_QISKIT_DEFAULT_SHOTS 512
#define QALLOW_MAX_QISKIT_SAMPLES 32

#ifdef _WIN32

static bool qallow_qiskit_enabled(void) {
    return false;
}

static void qallow_apply_qiskit_feedback(qallow_state_t* state) {
    (void)state;
}

#else

static qallow_run_metrics_t g_last_run_metrics = {0};

const qallow_run_metrics_t* qallow_get_last_run_metrics(void) {
    return &g_last_run_metrics;
}

void qallow_metrics_begin_run(int cuda_enabled) {
    memset(&g_last_run_metrics, 0, sizeof(g_last_run_metrics));
    g_last_run_metrics.cuda_enabled = cuda_enabled ? 1 : 0;
    g_last_run_metrics.equilibrium_tick = -1;
}

void qallow_metrics_update_tick(int tick, float coherence, float decoherence, int cuda_enabled) {
    g_last_run_metrics.tick_count = tick;
    g_last_run_metrics.final_coherence = coherence;
    g_last_run_metrics.final_decoherence = decoherence;
    g_last_run_metrics.cuda_enabled = cuda_enabled ? 1 : 0;
}

void qallow_metrics_mark_equilibrium(int tick) {
    if (tick < 0) {
        return;
    }
    g_last_run_metrics.reached_equilibrium = 1;
    if (g_last_run_metrics.equilibrium_tick < 0) {
        g_last_run_metrics.equilibrium_tick = tick;
    }
}

void qallow_metrics_finalize(float coherence, float decoherence) {
    g_last_run_metrics.final_coherence = coherence;
    g_last_run_metrics.final_decoherence = decoherence;
}

static bool qallow_env_truthy(const char* value) {
    if (!value) return false;
    if (strcmp(value, "1") == 0) return true;
    if (strcmp(value, "true") == 0) return true;
    if (strcmp(value, "TRUE") == 0) return true;
    if (strcmp(value, "yes") == 0) return true;
    if (strcmp(value, "YES") == 0) return true;
    return false;
}

static const char* qallow_get_qiskit_script(void) {
    const char* override_path = getenv("QALLOW_QISKIT_BRIDGE");
    if (override_path && override_path[0] != '\0') {
        return override_path;
    }
    return "scripts/qiskit_bridge.py";
}

static const char* qallow_token_from_value(float value) {
    if (value >= 0.75f) {
        return "1";
    }
    if (value <= 0.25f) {
        return "-1";
    }
    if (fabsf(value - 0.5f) <= 0.05f) {
        return "0";
    }
    return "N";
}

static bool qallow_qiskit_enabled(void) {
    const char* toggle = getenv("QALLOW_QISKIT");
    if (!qallow_env_truthy(toggle)) {
        return false;
    }

    const char* script_path = qallow_get_qiskit_script();
    errno = 0;
    if (access(script_path, R_OK) != 0) {
        static bool warned = false;
        if (!warned) {
            fprintf(stderr, "[Qallow][Qiskit] Bridge script not readable at %s (errno=%d)\n", script_path, errno);
            warned = true;
        }
        return false;
    }

    return true;
}

static bool qallow_run_qiskit_bridge(const float* values, int count, float* coherence_out) {
    if (!values || !coherence_out || count <= 0) {
        return false;
    }

    int sample = count;
    if (sample > QALLOW_MAX_QISKIT_SAMPLES) {
        sample = QALLOW_MAX_QISKIT_SAMPLES;
    }

    char states[256];
    states[0] = '\0';
    size_t offset = 0;

    for (int i = 0; i < sample; ++i) {
        const char* token = qallow_token_from_value(values[i]);
        int written = snprintf(states + offset, sizeof(states) - offset, i == 0 ? "%s" : ",%s", token);
        if (written < 0 || (size_t)written >= sizeof(states) - offset) {
            fprintf(stderr, "[Qallow][Qiskit] Failed to compose state token list\n");
            return false;
        }
        offset += (size_t)written;
    }

    const char* script = qallow_get_qiskit_script();
    char command[1024];
    int command_len = snprintf(command, sizeof(command),
                               "python3 \"%s\" --states \"%s\" --shots %d --allow-simulator",
                               script, states, QALLOW_QISKIT_DEFAULT_SHOTS);
    if (command_len < 0 || command_len >= (int)sizeof(command)) {
        fprintf(stderr, "[Qallow][Qiskit] Command buffer too small for bridge invocation\n");
        return false;
    }

    FILE* pipe = popen(command, "r");
    if (!pipe) {
        fprintf(stderr, "[Qallow][Qiskit] popen failed for bridge command: %s\n", command);
        return false;
    }

    char buffer[256];
    double coherence = 0.0;
    bool parsed = false;

    while (fgets(buffer, sizeof(buffer), pipe)) {
        if (sscanf(buffer, "coherence=%lf", &coherence) == 1) {
            parsed = true;
            break;
        }
    }

    int status = pclose(pipe);
    if (status == -1) {
        fprintf(stderr, "[Qallow][Qiskit] Failed to close bridge pipe\n");
        return false;
    }

    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : status;
        fprintf(stderr, "[Qallow][Qiskit] Bridge script exited abnormally (status=%d)\n", exit_code);
        return false;
    }

    if (!parsed) {
        fprintf(stderr, "[Qallow][Qiskit] Bridge output missing coherence token\n");
        return false;
    }

    *coherence_out = (float)coherence;
    return true;
}

static void qallow_apply_qiskit_feedback(qallow_state_t* state) {
    if (!state) {
        return;
    }

    if (!qallow_qiskit_enabled()) {
        return;
    }

    overlay_t* overlay = &state->overlays[OVERLAY_MYCELIAL];
    int nodes = overlay->node_count;
    if (nodes <= 0) {
        return;
    }

    int sample = nodes;
    if (sample > QALLOW_MAX_QISKIT_SAMPLES) {
        sample = QALLOW_MAX_QISKIT_SAMPLES;
    }

    float coherence = 0.0f;
    if (!qallow_run_qiskit_bridge(overlay->values, sample, &coherence)) {
        return;
    }

    for (int i = 0; i < sample; ++i) {
        overlay->history[i] = overlay->values[i];
        float blended = overlay->values[i] + (coherence - overlay->values[i]) * 0.25f;
        if (blended < 0.0f) {
            blended = 0.0f;
        }
        if (blended > 1.0f) {
            blended = 1.0f;
        }
        overlay->values[i] = blended;
    }

    fprintf(stdout, "[Qallow][Qiskit] coherence=%.4f (samples=%d)\n", coherence, sample);
}

#endif  // _WIN32

// Qallow Kernel - Core VM implementation

void qallow_kernel_init(qallow_state_t* state) {
    if (!state) return;

    memset(state, 0, sizeof(qallow_state_t));

    state->tick_count = 0;
    state->global_coherence = 0.5f;
    state->decoherence_level = 0.0f;
    state->cuda_enabled = false;
    state->gpu_device_id = 0;

    // Phase 8-10: Initialize ethics components
    state->ethics_S = 0.5f;  // Safety
    state->ethics_C = 0.5f;  // Clarity
    state->ethics_H = 0.5f;  // Human benefit

    // Initialize overlays
    for (int i = 0; i < NUM_OVERLAYS; i++) {
        state->overlays[i].node_count = MAX_NODES;
        state->overlays[i].stability = 0.5f;

        // Initialize node values
        for (int j = 0; j < MAX_NODES; j++) {
            state->overlays[i].values[j] = 0.5f + (float)rand() / RAND_MAX * 0.1f;
            state->overlays[i].history[j] = state->overlays[i].values[j];
        }
    }
    
#if CUDA_ENABLED
    // Try to initialize CUDA
    int device_count = 0;
    cudaError_t get_count_err = cudaGetDeviceCount(&device_count);
    if (get_count_err != cudaSuccess) {
        fprintf(stderr, "[CUDA] cudaGetDeviceCount failed: %s\n", cudaGetErrorString(get_count_err));
    } else if (device_count > 0) {
        state->cuda_enabled = true;
        qallow_cuda_init(state);
    }
    else {
        fprintf(stderr, "[CUDA] No CUDA devices detected.\n");
    }
#endif

    phase14_initialize(state);
    phase15_initialize(state);
}

void qallow_kernel_tick(qallow_state_t* state) {
    if (!state) return;

    state->tick_count++;

#if CUDA_ENABLED
    if (state->cuda_enabled) {
        qallow_cuda_process_overlays(state);
    } else {
        qallow_cpu_process_overlays(state);
    }
#else
    qallow_cpu_process_overlays(state);
#endif

    qallow_apply_qiskit_feedback(state);

    // Update decoherence
    qallow_update_decoherence(state);

    // Update global coherence
    float total_stability = 0.0f;
    for (int i = 0; i < NUM_OVERLAYS; i++) {
        state->overlays[i].stability = qallow_calculate_stability(&state->overlays[i]);
        total_stability += state->overlays[i].stability;
    }
    state->global_coherence = total_stability / NUM_OVERLAYS;

    // ====================================================================
    // Phase 8-10: Adaptive-Predictive-Temporal Loop
    // ====================================================================
    double pred = foresight_predict(qallow_global_stability(state));
    predictive_control(state);
    adaptive_governance(state);
    temporal_alignment(state, pred, qallow_global_stability(state));

    phase14_tick(state);
    phase15_tick(state);
}

// qallow_calculate_stability moved to header as inline function

CUDA_CALLABLE void qallow_update_decoherence(qallow_state_t* state) {
    if (!state) return;
    
    // Decoherence increases slightly each tick
    state->decoherence_level += 0.00001f;
    
    // Cap at maximum
    if (state->decoherence_level > 0.1f) {
        state->decoherence_level = 0.1f;
    }
    
    // Decoherence reduces coherence
    state->global_coherence *= (1.0f - state->decoherence_level * 0.001f);
}

void qallow_print_status(const qallow_state_t* state, int tick) {
    if (!state) return;
    
    printf("[TICK %04d] Coherence: %.4f | Decoherence: %.6f | Stability: ",
           tick, state->global_coherence, state->decoherence_level);
    
    for (int i = 0; i < NUM_OVERLAYS; i++) {
        printf("%.4f ", state->overlays[i].stability);
    }
    printf("\n");
}

bool qallow_ethics_check(const qallow_state_t* state, ethics_state_t* ethics) {
    if (!state || !ethics) return false;
    
    // Use the full ethics module for proper evaluation
    static ethics_monitor_t ethics_monitor;
    static bool ethics_initialized = false;
    
    if (!ethics_initialized) {
        ethics_init(&ethics_monitor);
        ethics_initialized = true;
    }
    
    // Evaluate using the full ethics module
    bool passed = ethics_evaluate_state(state, &ethics_monitor);
    
    // Copy results to the simple ethics_state_t structure
    ethics->safety_score = ethics_calculate_safety_score(state, &ethics_monitor);
    ethics->clarity_score = ethics_calculate_clarity_score(state, &ethics_monitor);
    ethics->human_benefit_score = ethics_calculate_human_benefit_score(state, &ethics_monitor);
    ethics->total_ethics_score = ethics_monitor.total_ethics_score;
    ethics->safety_check_passed = passed;
    
    return passed;
}

void qallow_cpu_process_overlays(qallow_state_t* state) {
    if (!state) return;
    
    // CPU-based overlay processing
    for (int overlay_idx = 0; overlay_idx < NUM_OVERLAYS; overlay_idx++) {
        overlay_t* overlay = &state->overlays[overlay_idx];
        
        // Simple diffusion-like update
        for (int i = 0; i < overlay->node_count; i++) {
            float new_val = overlay->values[i];
            
            // Add small random perturbation
            new_val += (float)rand() / RAND_MAX * 0.01f - 0.005f;
            
            // Clamp to [0, 1]
            if (new_val < 0.0f) new_val = 0.0f;
            if (new_val > 1.0f) new_val = 1.0f;
            
            overlay->history[i] = overlay->values[i];
            overlay->values[i] = new_val;
        }
    }
}

#if CUDA_ENABLED
void qallow_cuda_init(qallow_state_t* state) {
    if (!state) return;
    state->gpu_device_id = 0;
    cudaSetDevice(state->gpu_device_id);
}

void qallow_cuda_cleanup(qallow_state_t* state) {
    if (!state) return;
    (void)state;
    cudaDeviceSynchronize();
}

void qallow_cuda_process_overlays(qallow_state_t* state) {
    if (!state) return;
    
    double buffer[MAX_NODES];

    overlay_t* orbital = &state->overlays[OVERLAY_ORBITAL];
    int nodes = orbital->node_count;
    if (nodes > MAX_NODES) nodes = MAX_NODES;
    for (int i = 0; i < nodes; i++) {
        orbital->history[i] = orbital->values[i];
        buffer[i] = orbital->values[i];
    }
    runPhotonicSimulation(buffer, nodes, (unsigned long)(state->tick_count + 1));
    for (int i = 0; i < nodes; i++) {
        orbital->values[i] = (float)buffer[i];
    }

    overlay_t* river = &state->overlays[OVERLAY_RIVER_DELTA];
    nodes = river->node_count;
    if (nodes > MAX_NODES) nodes = MAX_NODES;
    for (int i = 0; i < nodes; i++) {
        river->history[i] = river->values[i];
        buffer[i] = river->values[i];
    }
    runQuantumOptimizer(buffer, nodes);
    for (int i = 0; i < nodes; i++) {
        river->values[i] = (float)buffer[i];
    }

    overlay_t* mycelial = &state->overlays[OVERLAY_MYCELIAL];
    nodes = mycelial->node_count;
    if (nodes > MAX_NODES) nodes = MAX_NODES;
    for (int i = 0; i < nodes; i++) {
        mycelial->history[i] = mycelial->values[i];
        buffer[i] = mycelial->values[i];
    }
    runQuantumOptimizer(buffer, nodes);
    for (int i = 0; i < nodes; i++) {
        mycelial->values[i] = (float)buffer[i];
    }

    overlay_apply_interactions(state->overlays, NUM_OVERLAYS);
}
#endif

// ============================================================================
// ASCII DASHBOARD FUNCTIONS
// ============================================================================

void qallow_print_bar(const char* label, double value, int width) {
    if (value < 0.0) value = 0.0;
    if (value > 1.0) value = 1.0;
    
    int filled = (int)lrint(value * width);
    printf("%-12s | ", label);
    for (int i = 0; i < width; i++) {
        putchar(i < filled ? '#' : '.');
    }
    printf(" | %.4f\n", value);
}

void qallow_print_dashboard(const qallow_state_t* state, const ethics_state_t* ethics) {
    if (!state) return;
    
    printf("\n╔════════════════════════════════════════════════════════════╗\n");
    printf("║           Qallow VM Dashboard - Tick %-6d             ║\n", state->tick_count);
    printf("╚════════════════════════════════════════════════════════════╝\n\n");
    
    // Overlay stability bars
    printf("OVERLAY STABILITY:\n");
    qallow_print_bar("Orbital", state->overlays[OVERLAY_ORBITAL].stability, 40);
    qallow_print_bar("River", state->overlays[OVERLAY_RIVER_DELTA].stability, 40);
    qallow_print_bar("Mycelial", state->overlays[OVERLAY_MYCELIAL].stability, 40);
    qallow_print_bar("Global", state->global_coherence, 40);
    printf("\n");
    
    // Ethics components
    if (ethics) {
        printf("ETHICS MONITORING:\n");
        qallow_print_bar("Safety (S)", ethics->safety_score, 40);
        qallow_print_bar("Clarity (C)", ethics->clarity_score, 40);
        qallow_print_bar("Human (H)", ethics->human_benefit_score, 40);
        printf("%-12s   E = S+C+H = %.2f (Safety=%.2f, Clarity=%.2f, Human=%.2f)\n",
               "", ethics->total_ethics_score, 
               ethics->safety_score, ethics->clarity_score, ethics->human_benefit_score);
        printf("%-12s   Status: %s\n\n", "", 
               ethics->safety_check_passed ? "PASS ✓" : "FAIL ✗");
    }
    
    // Decoherence (inverted bar for coherence visualization)
    printf("COHERENCE:\n");
    double coherence_bar = 1.0 - fmin(fmax(state->decoherence_level / 0.1, 0.0), 1.0);
    qallow_print_bar("Coherence", coherence_bar, 40);
    printf("%-12s   decoherence = %.6f\n", "", state->decoherence_level);
    printf("%-12s   Mode: %s\n\n", "", state->cuda_enabled ? "CUDA GPU" : "CPU");
}

// ============================================================================
// CSV LOGGING FUNCTIONS
// ============================================================================

static FILE* csv_log_file = NULL;

void qallow_csv_log_init(const char* filepath) {
    if (!filepath) return;
    
    csv_log_file = fopen(filepath, "w");
    if (!csv_log_file) {
        fprintf(stderr, "[CSV] Warning: Could not open %s for logging\n", filepath);
        return;
    }
    
    // Write CSV header
    fprintf(csv_log_file, "tick,orbital,river,mycelial,global,decoherence,ethics_S,ethics_C,ethics_H,ethics_total,ethics_pass\n");
    fflush(csv_log_file);
    
    printf("[CSV] Logging enabled: %s\n", filepath);
}

void qallow_csv_log_tick(const qallow_state_t* state, const ethics_state_t* ethics) {
    if (!csv_log_file || !state) return;
    
    fprintf(csv_log_file, "%d,%.6f,%.6f,%.6f,%.6f,%.6f",
            state->tick_count,
            state->overlays[OVERLAY_ORBITAL].stability,
            state->overlays[OVERLAY_RIVER_DELTA].stability,
            state->overlays[OVERLAY_MYCELIAL].stability,
            state->global_coherence,
            state->decoherence_level);
    
    if (ethics) {
        fprintf(csv_log_file, ",%.6f,%.6f,%.6f,%.6f,%d",
                ethics->safety_score,
                ethics->clarity_score,
                ethics->human_benefit_score,
                ethics->total_ethics_score,
                ethics->safety_check_passed ? 1 : 0);
    } else {
        fprintf(csv_log_file, ",0,0,0,0,0");
    }
    
    fprintf(csv_log_file, "\n");
    fflush(csv_log_file);
}

void qallow_csv_log_close(void) {
    if (csv_log_file) {
        fclose(csv_log_file);
        csv_log_file = NULL;
        printf("[CSV] Log closed\n");
    }
}

// ========================================================================
// Phase 8-10: Adaptive-Predictive-Temporal Loop
// ========================================================================

/* Helper: Calculate global stability from all overlays */
float qallow_global_stability(const qallow_state_t* state) {
    if (!state) return 0.0f;
    float total = 0.0f;
    for (int i = 0; i < NUM_OVERLAYS; i++) {
        total += state->overlays[i].stability;
    }
    return total / NUM_OVERLAYS;
}

/* Phase 8: Adaptive Governance — maintain ethics balance */
void adaptive_governance(qallow_state_t* state) {
    if (!state) return;

    double g = qallow_global_stability(state);
    double err = 0.995 - g;  // target stability

    state->ethics_S += 0.10f * (float)err;
    state->ethics_C += 0.05f * (float)err;
    state->ethics_H += 0.05f * (float)err;

    // keep within [0,1]
    if (state->ethics_S < 0.0f) state->ethics_S = 0.0f;
    if (state->ethics_S > 1.0f) state->ethics_S = 1.0f;
    if (state->ethics_C < 0.0f) state->ethics_C = 0.0f;
    if (state->ethics_C > 1.0f) state->ethics_C = 1.0f;
    if (state->ethics_H < 0.0f) state->ethics_H = 0.0f;
    if (state->ethics_H > 1.0f) state->ethics_H = 1.0f;

    // damp decoherence when unstable
    if (g < 0.990f) {
        state->decoherence_level *= 0.98f;
    }
}

/* Phase 9: Predictive Control — forecast next-tick stability */
#define QALLOW_WINDOW 8
typedef struct {
    double h[QALLOW_WINDOW];
    int i;
} foresight_t;
static foresight_t foresight_global = {{0}, 0};

double foresight_predict(double now) {
    double prev = foresight_global.h[(foresight_global.i - 1 + QALLOW_WINDOW) % QALLOW_WINDOW];
    foresight_global.h[foresight_global.i] = now;
    foresight_global.i = (foresight_global.i + 1) % QALLOW_WINDOW;
    return now + (now - prev);  // one-step extrapolation
}

void predictive_control(qallow_state_t* state) {
    if (!state) return;

    double now = qallow_global_stability(state);
    double pred = foresight_predict(now);
    double err = pred - now;

    if (fabs(err) > 0.002) {
        state->decoherence_level *= (float)(err > 0 ? 0.98 : 1.02);
        state->ethics_S += 0.02f * (float)err;
        state->ethics_C += 0.01f * (float)err;
        state->ethics_H += 0.01f * (float)err;
    }

    // clamp
    if (state->ethics_S < 0.0f) state->ethics_S = 0.0f;
    if (state->ethics_S > 1.0f) state->ethics_S = 1.0f;
    if (state->ethics_C < 0.0f) state->ethics_C = 0.0f;
    if (state->ethics_C > 1.0f) state->ethics_C = 1.0f;
    if (state->ethics_H < 0.0f) state->ethics_H = 0.0f;
    if (state->ethics_H > 1.0f) state->ethics_H = 1.0f;
}

/* Phase 10: Temporal Memory Alignment — validate prediction history */
typedef struct {
    double mae;        // mean absolute error
    double total_err;
    unsigned long n;
} temporal_state_t;
static temporal_state_t temporal_state = {0};

void temporal_alignment(qallow_state_t* state, double predicted, double actual) {
    if (!state) return;

    double e = fabs(predicted - actual);
    temporal_state.total_err += e;
    temporal_state.n++;
    temporal_state.mae = temporal_state.total_err / temporal_state.n;

    // if drift too high, tighten decoherence and ethics weights
    if (temporal_state.mae > 0.003) {
        double adj = 1.0 - fmin(temporal_state.mae * 50.0, 0.1);
        state->decoherence_level *= (float)adj;
        state->ethics_S *= (float)adj;
        state->ethics_C *= (float)adj;
        state->ethics_H *= (float)adj;
    }
}
