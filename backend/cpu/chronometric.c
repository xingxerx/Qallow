#include "chronometric.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Chronometric Prediction Layer Implementation
// Links simulation ticks to temporal forecast indices

// Initialize chronometric time bank
void chrono_bank_init(chrono_bank_t* bank, float learning_rate, float decay_factor) {
    if (!bank) return;
    
    memset(bank, 0, sizeof(chrono_bank_t));
    
    bank->learning_rate = learning_rate;
    bank->decay_factor = decay_factor;
    bank->confidence_threshold = 0.7f;
    
    printf("[CHRONO-BANK] Initialized - LR=%.4f Decay=%.4f\n", 
           learning_rate, decay_factor);
}

// Add observation to time bank
void chrono_bank_add_observation(chrono_bank_t* bank,
                                 double delta_t,
                                 double confidence,
                                 int event_id) {
    if (!bank) return;
    
    // Get next entry position
    int idx = bank->entry_count % CHRONO_HISTORY_SIZE;
    
    // Store entry
    bank->history[idx].delta_t = delta_t;
    bank->history[idx].confidence = confidence;
    bank->history[idx].timestamp = (double)clock() / CLOCKS_PER_SEC;
    bank->history[idx].event_id = event_id;
    
    bank->entry_count++;
    
    // Update running statistics
    int n = (bank->entry_count < CHRONO_HISTORY_SIZE) ? 
            bank->entry_count : CHRONO_HISTORY_SIZE;
    
    // Incremental mean and variance update
    double old_mean = bank->avg_delta_t;
    bank->avg_delta_t += (delta_t - bank->avg_delta_t) / n;
    
    double variance = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = bank->history[i].delta_t - bank->avg_delta_t;
        variance += diff * diff;
    }
    bank->std_delta_t = sqrt(variance / n);
    
    // Update confidence
    double conf_sum = 0.0;
    for (int i = 0; i < n; i++) {
        conf_sum += bank->history[i].confidence;
    }
    bank->overall_confidence = conf_sum / n;
}

// Get predicted temporal offset
double chrono_bank_predict_delta_t(const chrono_bank_t* bank) {
    if (!bank || bank->entry_count == 0) return 0.0;
    
    // Weighted prediction based on recent history
    int n = (bank->entry_count < CHRONO_HISTORY_SIZE) ? 
            bank->entry_count : CHRONO_HISTORY_SIZE;
    
    double weighted_sum = 0.0;
    double weight_total = 0.0;
    
    for (int i = 0; i < n; i++) {
        int idx = (bank->entry_count - 1 - i) % CHRONO_HISTORY_SIZE;
        
        // Recent observations get higher weight
        double recency_weight = pow(bank->decay_factor, i);
        double confidence_weight = bank->history[idx].confidence;
        double total_weight = recency_weight * confidence_weight;
        
        weighted_sum += bank->history[idx].delta_t * total_weight;
        weight_total += total_weight;
    }
    
    return (weight_total > 0.0) ? (weighted_sum / weight_total) : 0.0;
}

// Get confidence in prediction
double chrono_bank_get_confidence(const chrono_bank_t* bank) {
    if (!bank) return 0.0;
    
    // Confidence based on:
    // 1. Overall confidence of observations
    // 2. Consistency (inverse of std dev)
    // 3. Sample size
    
    double sample_confidence = (bank->entry_count >= CHRONO_HISTORY_SIZE) ? 
                              1.0 : (double)bank->entry_count / CHRONO_HISTORY_SIZE;
    
    double consistency = 1.0 / (1.0 + bank->std_delta_t);
    
    return bank->overall_confidence * consistency * sample_confidence;
}

// Print bank statistics
void chrono_bank_print_stats(const chrono_bank_t* bank) {
    if (!bank) return;
    
    printf("\n=== CHRONOMETRIC TIME BANK ===\n");
    printf("Observations:        %d\n", bank->entry_count);
    printf("Avg Δt:              %.6f sec\n", bank->avg_delta_t);
    printf("Std Dev:             %.6f sec\n", bank->std_delta_t);
    printf("Overall Confidence:  %.4f\n", bank->overall_confidence);
    printf("Prediction:          %.6f sec\n", chrono_bank_predict_delta_t(bank));
    printf("Prediction Conf:     %.4f\n", chrono_bank_get_confidence(bank));
    printf("==============================\n\n");
}

// Initialize chronometric state
void chronometric_init(chronometric_state_t* chrono, 
                      float learning_rate,
                      float decay_factor) {
    if (!chrono) return;
    
    memset(chrono, 0, sizeof(chronometric_state_t));
    
    chrono_bank_init(&chrono->time_bank, learning_rate, decay_factor);
    
    chrono->drift_threshold = 0.01;  // 10ms drift threshold
    chrono->drift_alert_active = false;
    
    // Open telemetry file
    chrono->telemetry_file = fopen("chronometric_telemetry.csv", "w");
    if (chrono->telemetry_file) {
        fprintf(chrono->telemetry_file, 
                "tick,delta_t,predicted_delta_t,drift,confidence,alert\n");
    }
    
    printf("[CHRONOMETRIC] State initialized\n");
}

// Cleanup chronometric state
void chronometric_cleanup(chronometric_state_t* chrono) {
    if (!chrono) return;
    
    if (chrono->telemetry_file) {
        fclose(chrono->telemetry_file);
    }
    
    printf("[CHRONOMETRIC] Cleanup complete\n");
}

// Update chronometric state with new observation
void chronometric_update(chronometric_state_t* chrono,
                        int tick,
                        double observed_time,
                        double predicted_time,
                        const qallow_state_t* state) {
    if (!chrono || !state) return;
    
    // Calculate delta_t
    double delta_t = observed_time - predicted_time;
    
    // Confidence based on system coherence
    double confidence = state->global_coherence;
    
    // Add to time bank
    chrono_bank_add_observation(&chrono->time_bank, delta_t, confidence, tick);
    
    // Update drift tracking
    chrono->accumulated_drift += delta_t;
    chrono->drift_rate = chrono->accumulated_drift / (tick + 1);
    
    // Check for drift alert
    if (fabs(delta_t) > chrono->drift_threshold) {
        chrono->drift_alert_active = true;
        printf("[CHRONO-ALERT] Tick %d: Temporal drift detected! Δt=%.6f sec\n",
               tick, delta_t);
    } else {
        chrono->drift_alert_active = false;
    }
    
    // Write telemetry
    if (chrono->telemetry_file) {
        double predicted_delta = chrono_bank_predict_delta_t(&chrono->time_bank);
        fprintf(chrono->telemetry_file, "%d,%.6f,%.6f,%.6f,%.4f,%d\n",
               tick, delta_t, predicted_delta, chrono->accumulated_drift,
               confidence, chrono->drift_alert_active ? 1 : 0);
    }
}

// Generate temporal forecast
void chronometric_generate_forecast(chronometric_state_t* chrono,
                                   int current_tick,
                                   const qallow_state_t* current_state) {
    if (!chrono || !current_state) return;
    
    double predicted_delta = chrono_bank_predict_delta_t(&chrono->time_bank);
    double forecast_confidence = chrono_bank_get_confidence(&chrono->time_bank);
    
    // Generate forecasts for future ticks
    for (int i = 0; i < CHRONO_FORECAST_HORIZON; i++) {
        temporal_forecast_t* forecast = &chrono->forecasts[i];
        
        forecast->tick_offset = i + 1;
        forecast->predicted_time = predicted_delta * (i + 1);
        
        // Predict metrics with decay
        float decay = pow(0.98f, i);  // Confidence decays with distance
        forecast->predicted_coherence = current_state->global_coherence * decay;
        forecast->predicted_decoherence = current_state->decoherence_level * (1.0f + i * 0.01f);
        forecast->predicted_ethics = current_state->overlays[0].stability * decay;
        
        forecast->confidence = forecast_confidence * decay;
    }
    
    chrono->forecast_generation_tick = current_tick;
}

// Get forecast for specific tick offset
const temporal_forecast_t* chronometric_get_forecast(const chronometric_state_t* chrono,
                                                    int tick_offset) {
    if (!chrono || tick_offset < 1 || tick_offset > CHRONO_FORECAST_HORIZON) {
        return NULL;
    }
    
    return &chrono->forecasts[tick_offset - 1];
}

// Update forecast based on actual observation
void chronometric_update_forecast(chronometric_state_t* chrono,
                                 int tick,
                                 const qallow_state_t* actual_state) {
    if (!chrono || !actual_state) return;
    
    // Find corresponding forecast
    int offset = tick - chrono->forecast_generation_tick;
    if (offset < 1 || offset > CHRONO_FORECAST_HORIZON) return;
    
    const temporal_forecast_t* forecast = &chrono->forecasts[offset - 1];
    
    // Calculate forecast errors
    float coherence_error = fabs(forecast->predicted_coherence - actual_state->global_coherence);
    float decoherence_error = fabs(forecast->predicted_decoherence - actual_state->decoherence_level);
    
    // Store for learning
    chrono->last_forecast_error = (coherence_error + decoherence_error) / 2.0f;
    
    printf("[CHRONO-FORECAST] Tick %d offset %d: Error=%.4f\n",
           tick, offset, chrono->last_forecast_error);
}

// Track drift over time
void chronometric_track_drift(chronometric_state_t* chrono, 
                             double current_time,
                             double expected_time) {
    if (!chrono) return;
    
    chrono->last_drift_check_time = current_time;
    chrono->last_expected_time = expected_time;
    
    double instant_drift = current_time - expected_time;
    
    if (fabs(instant_drift) > chrono->drift_threshold) {
        printf("[CHRONO-DRIFT] Drift detected: %.6f sec (threshold: %.6f)\n",
               instant_drift, chrono->drift_threshold);
    }
}

// Detect temporal anomalies
bool chronometric_detect_anomaly(const chronometric_state_t* chrono,
                                double delta_t) {
    if (!chrono || chrono->time_bank.entry_count < 10) return false;
    
    // Anomaly if delta_t is more than 3 sigma from mean
    double z_score = fabs(delta_t - chrono->time_bank.avg_delta_t) / 
                    (chrono->time_bank.std_delta_t + 1e-10);
    
    return (z_score > 3.0);
}

// Calculate temporal offset for tick
double chronometric_calculate_offset(const chronometric_state_t* chrono,
                                    int tick) {
    if (!chrono) return 0.0;
    
    // Base offset from time bank prediction
    double base_offset = chrono_bank_predict_delta_t(&chrono->time_bank);
    
    // Add drift correction
    double drift_correction = chrono->drift_rate * tick;
    
    return base_offset + drift_correction;
}

// Analyze temporal patterns
void chronometric_analyze_patterns(const chronometric_state_t* chrono) {
    if (!chrono || chrono->time_bank.entry_count < CHRONO_HISTORY_SIZE) {
        printf("[CHRONO-PATTERN] Insufficient data for pattern analysis\n");
        return;
    }
    
    printf("\n=== TEMPORAL PATTERN ANALYSIS ===\n");
    
    // Analyze periodicity
    int n = CHRONO_HISTORY_SIZE;
    double autocorr = 0.0;
    double mean = chrono->time_bank.avg_delta_t;
    
    for (int lag = 1; lag < 20; lag++) {
        double corr = 0.0;
        for (int i = 0; i < n - lag; i++) {
            double x = chrono->time_bank.history[i].delta_t - mean;
            double y = chrono->time_bank.history[i + lag].delta_t - mean;
            corr += x * y;
        }
        corr /= (n - lag);
        
        if (fabs(corr) > 0.5) {
            printf("  Potential period at lag %d: correlation=%.4f\n", lag, corr);
        }
    }
    
    // Trend analysis
    double trend = (chrono->time_bank.history[n-1].delta_t - 
                   chrono->time_bank.history[0].delta_t) / n;
    printf("  Trend: %.6f sec/observation\n", trend);
    
    // Drift summary
    printf("  Accumulated drift: %.6f sec\n", chrono->accumulated_drift);
    printf("  Drift rate: %.6f sec/tick\n", chrono->drift_rate);
    
    printf("==================================\n\n");
}

// Print forecast
void chronometric_print_forecast(const chronometric_state_t* chrono, int horizon) {
    if (!chrono) return;
    
    if (horizon > CHRONO_FORECAST_HORIZON) horizon = CHRONO_FORECAST_HORIZON;
    
    printf("\n=== TEMPORAL FORECAST ===\n");
    printf("Offset | Time    | Coherence | Decohere | Confidence\n");
    printf("-------|---------|-----------|----------|------------\n");
    
    for (int i = 0; i < horizon; i++) {
        const temporal_forecast_t* f = &chrono->forecasts[i];
        printf(" +%2d   | %7.4f |   %.4f   |  %.6f  |   %.4f\n",
               f->tick_offset, f->predicted_time, f->predicted_coherence,
               f->predicted_decoherence, f->confidence);
    }
    
    printf("=========================\n\n");
}

// Write telemetry summary
void chronometric_write_summary(const chronometric_state_t* chrono) {
    if (!chrono) return;
    
    FILE* f = fopen("chronometric_summary.txt", "w");
    if (!f) return;
    
    fprintf(f, "Chronometric Prediction Summary\n");
    fprintf(f, "================================\n\n");
    
    fprintf(f, "Time Bank Statistics:\n");
    fprintf(f, "  Observations: %d\n", chrono->time_bank.entry_count);
    fprintf(f, "  Avg Δt: %.6f sec\n", chrono->time_bank.avg_delta_t);
    fprintf(f, "  Std Dev: %.6f sec\n", chrono->time_bank.std_delta_t);
    fprintf(f, "  Confidence: %.4f\n\n", chrono->time_bank.overall_confidence);
    
    fprintf(f, "Drift Tracking:\n");
    fprintf(f, "  Accumulated: %.6f sec\n", chrono->accumulated_drift);
    fprintf(f, "  Rate: %.6f sec/tick\n", chrono->drift_rate);
    fprintf(f, "  Alert Active: %s\n\n", chrono->drift_alert_active ? "YES" : "NO");
    
    fprintf(f, "Forecast Horizon: %d ticks\n", CHRONO_FORECAST_HORIZON);
    fprintf(f, "Last Forecast Error: %.6f\n", chrono->last_forecast_error);
    
    fclose(f);
    printf("[CHRONOMETRIC] Summary written to chronometric_summary.txt\n");
}