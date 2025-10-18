#include "chronometric.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Chronometric Prediction Layer - Corrected Implementation
// Matches the interface in chronometric.h

// Initialize chronometric time bank
void chrono_bank_init(chrono_bank_t* bank) {
    if (!bank) return;
    
    memset(bank, 0, sizeof(chrono_bank_t));
    
    bank->learning_rate = 0.01;
    bank->confidence_decay = 0.95;
    bank->adaptation_threshold = 0.1;
    bank->history_count = 0;
    bank->current_index = 0;
    
    printf("[CHRONO-BANK] Initialized\n");
}

// Record an event in the time bank
void chrono_bank_record_event(chrono_bank_t* bank,
                              int event_id,
                              double observed_time,
                              double predicted_time) {
    if (!bank) return;
    
    int idx = bank->current_index;
    
    bank->history[idx].event_id = event_id;
    bank->history[idx].delta_t = observed_time - predicted_time;
    bank->history[idx].timestamp = observed_time;
    bank->history[idx].confidence = 1.0 - fabs(bank->history[idx].delta_t);
    
    bank->current_index = (bank->current_index + 1) % CHRONO_HISTORY_SIZE;
    if (bank->history_count < CHRONO_HISTORY_SIZE) {
        bank->history_count++;
    }
    
    chrono_bank_update_stats(bank);
}

// Update time bank statistics
void chrono_bank_update_stats(chrono_bank_t* bank) {
    if (!bank || bank->history_count == 0) return;
    
    int n = bank->history_count;
    
    // Calculate mean delta_t
    double sum = 0.0;
    double conf_sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += bank->history[i].delta_t;
        conf_sum += bank->history[i].confidence;
    }
    bank->avg_delta_t = sum / n;
    bank->overall_confidence = conf_sum / n;
    
    // Calculate standard deviation
    double variance = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = bank->history[i].delta_t - bank->avg_delta_t;
        variance += diff * diff;
    }
    bank->std_delta_t = sqrt(variance / n);
}

// Get overall confidence from time bank
double chrono_bank_get_confidence(const chrono_bank_t* bank) {
    if (!bank || bank->history_count == 0) return 0.0;
    
    // Confidence based on consistency and sample size
    double sample_factor = (double)bank->history_count / CHRONO_HISTORY_SIZE;
    double consistency = 1.0 / (1.0 + bank->std_delta_t);
    
    return bank->overall_confidence * consistency * sample_factor;
}

// Initialize chronometric state
void chronometric_init(chronometric_state_t* chrono) {
    if (!chrono) return;
    
    memset(chrono, 0, sizeof(chronometric_state_t));
    
    chrono_bank_init(&chrono->time_bank);
    
    chrono->simulation_start_time = (double)clock() / CLOCKS_PER_SEC;
    chrono->last_tick_time = chrono->simulation_start_time;
    chrono->drift_alert_active = false;
    
    // Open telemetry file
    sprintf(chrono->telemetry_file, "chronometric_telemetry.csv");
    chrono->chrono_telemetry = fopen(chrono->telemetry_file, "w");
    
    if (chrono->chrono_telemetry) {
        fprintf(chrono->chrono_telemetry,
                "tick,delta_t,predicted_delta_t,drift,confidence,alert\n");
    }
    
    printf("[CHRONOMETRIC] State initialized\n");
}

// Cleanup chronometric state
void chronometric_cleanup(chronometric_state_t* chrono) {
    if (!chrono) return;
    
    if (chrono->chrono_telemetry) {
        fclose(chrono->chrono_telemetry);
    }
    
    printf("[CHRONOMETRIC] Cleanup complete\n");
}

// Generate temporal forecasts
void chronometric_generate_forecasts(chronometric_state_t* chrono,
                                     const qallow_state_t* current_state,
                                     int horizon_ticks) {
    if (!chrono || !current_state) return;
    
    if (horizon_ticks > CHRONO_FORECAST_HORIZON) {
        horizon_ticks = CHRONO_FORECAST_HORIZON;
    }
    
    double predicted_delta = chrono->time_bank.avg_delta_t;
    double base_confidence = chrono_bank_get_confidence(&chrono->time_bank);
    
    for (int i = 0; i < horizon_ticks; i++) {
        temporal_forecast_t* forecast = &chrono->forecasts[i];
        
        forecast->tick_offset = i + 1;
        forecast->predicted_time = predicted_delta * (i + 1);
        
        // Predict metrics with decay
        double decay = pow(0.98, i);
        forecast->predicted_coherence = current_state->global_coherence * decay;
        forecast->predicted_decoherence = current_state->decoherence_level * (1.0 + i * 0.01);
        forecast->predicted_ethics = 2.99f * decay;
        
        forecast->confidence = base_confidence * decay;
    }
    
    chrono->num_forecasts = horizon_ticks;
}

// Update forecast based on observation
void chronometric_update_forecast(chronometric_state_t* chrono,
                                 const qallow_state_t* observed_state,
                                 int tick) {
    if (!chrono || !observed_state) return;
    
    // Record actual timing for this tick
    double current_time = (double)clock() / CLOCKS_PER_SEC;
    double tick_duration = current_time - chrono->last_tick_time;
    
    chrono_bank_record_event(&chrono->time_bank, tick,
                            current_time,
                            chrono->last_tick_time + chrono->tick_duration_avg);
    
    chrono->last_tick_time = current_time;
    chronometric_update_tick_timing(chrono, tick_duration);
}

// Track drift over time
void chronometric_track_drift(chronometric_state_t* chrono,
                             double observed_time,
                             double expected_time) {
    if (!chrono) return;
    
    double instant_drift = observed_time - expected_time;
    chrono->accumulated_drift += instant_drift;
    
    double elapsed = observed_time - chrono->simulation_start_time;
    if (elapsed > 0.0) {
        chrono->drift_rate = chrono->accumulated_drift / elapsed;
    }
    
    // Alert if drift exceeds 10ms
    if (fabs(instant_drift) > 0.01) {
        chrono->drift_alert_active = true;
        printf("[CHRONO-ALERT] Temporal drift: %.6f sec\n", instant_drift);
    } else {
        chrono->drift_alert_active = false;
    }
}

// Update tick timing statistics
void chronometric_update_tick_timing(chronometric_state_t* chrono,
                                    double tick_duration) {
    if (!chrono) return;
    
    if (chrono->tick_duration_avg == 0.0) {
        chrono->tick_duration_avg = tick_duration;
    } else {
        chrono->tick_duration_avg = chrono->tick_duration_avg * 0.95 + tick_duration * 0.05;
    }
    
    double diff = tick_duration - chrono->tick_duration_avg;
    chrono->tick_duration_std = sqrt(chrono->tick_duration_std * chrono->tick_duration_std * 0.95 + diff * diff * 0.05);
}

// Detect temporal anomalies
bool chronometric_detect_anomaly(const chronometric_state_t* chrono,
                                const qallow_state_t* state) {
    if (!chrono || !state) return false;
    if (chrono->time_bank.history_count < 10) return false;
    
    if (chrono->drift_alert_active) return true;
    if (state->decoherence_level > 0.1) return true;
    
    double current_time = (double)clock() / CLOCKS_PER_SEC;
    double elapsed = current_time - chrono->last_tick_time;
    
    if (elapsed > chrono->tick_duration_avg + 3.0 * chrono->tick_duration_std) {
        return true;
    }
    
    return false;
}

// Calculate temporal offset
double chronometric_calculate_temporal_offset(const chronometric_state_t* chrono) {
    if (!chrono) return 0.0;
    return chrono->time_bank.avg_delta_t + chrono->accumulated_drift;
}

// Predict next tick time
double chronometric_predict_next_tick_time(const chronometric_state_t* chrono) {
    if (!chrono) return 0.0;
    return chrono->last_tick_time + chrono->tick_duration_avg;
}

// Analyze temporal patterns
void chronometric_analyze_patterns(chronometric_state_t* chrono) {
    if (!chrono || chrono->time_bank.history_count < 20) {
        printf("[CHRONO-PATTERN] Insufficient data\n");
        return;
    }
    
    printf("\n=== TEMPORAL PATTERN ANALYSIS ===\n");
    printf("History count: %d\n", chrono->time_bank.history_count);
    printf("Avg delta-t: %.6f sec\n", chrono->time_bank.avg_delta_t);
    printf("Std dev: %.6f sec\n", chrono->time_bank.std_delta_t);
    printf("Confidence: %.4f\n", chrono_bank_get_confidence(&chrono->time_bank));
    printf("Accumulated drift: %.6f sec\n", chrono->accumulated_drift);
    printf("Drift rate: %.6f sec/sec\n", chrono->drift_rate);
    printf("Avg tick duration: %.6f sec\n", chrono->tick_duration_avg);
    printf("=================================\n\n");
}

// Write telemetry for current tick
void chronometric_write_telemetry(chronometric_state_t* chrono, int tick) {
    if (!chrono || !chrono->chrono_telemetry) return;
    
    double predicted_delta = chrono->time_bank.avg_delta_t;
    double confidence = chrono_bank_get_confidence(&chrono->time_bank);
    
    if (chrono->time_bank.history_count > 0) {
        int latest_idx = (chrono->time_bank.current_index - 1 + CHRONO_HISTORY_SIZE) % CHRONO_HISTORY_SIZE;
        double actual_delta = chrono->time_bank.history[latest_idx].delta_t;
        
        fprintf(chrono->chrono_telemetry, "%d,%.6f,%.6f,%.6f,%.4f,%d\n",
               tick, actual_delta, predicted_delta, chrono->accumulated_drift,
               confidence, chrono->drift_alert_active ? 1 : 0);
    }
}

// Write forecast report
void chronometric_write_forecast_report(const chronometric_state_t* chrono) {
    if (!chrono) return;
    
    FILE* f = fopen("chronometric_forecast.txt", "w");
    if (!f) return;
    
    fprintf(f, "Chronometric Forecast Report\n");
    fprintf(f, "=============================\n\n");
    
    fprintf(f, "Time Bank Statistics:\n");
    fprintf(f, "  Observations: %d\n", chrono->time_bank.history_count);
    fprintf(f, "  Avg delta-t: %.6f sec\n", chrono->time_bank.avg_delta_t);
    fprintf(f, "  Std dev: %.6f sec\n", chrono->time_bank.std_delta_t);
    fprintf(f, "  Confidence: %.4f\n\n", chrono_bank_get_confidence(&chrono->time_bank));
    
    fprintf(f, "Drift Tracking:\n");
    fprintf(f, "  Accumulated: %.6f sec\n", chrono->accumulated_drift);
    fprintf(f, "  Rate: %.6f sec/sec\n", chrono->drift_rate);
    fprintf(f, "  Alert: %s\n\n", chrono->drift_alert_active ? "ACTIVE" : "None");
    
    fprintf(f, "Forecasts (horizon=%d ticks):\n", chrono->num_forecasts);
    fprintf(f, "Offset | Time    | Coherence | Decohere | Confidence\n");
    fprintf(f, "-------|---------|-----------|----------|------------\n");
    
    for (int i = 0; i < chrono->num_forecasts && i < 20; i++) {
        const temporal_forecast_t* fc = &chrono->forecasts[i];
        fprintf(f, " +%2d   | %7.4f |   %.4f   |  %.6f  |   %.4f\n",
               fc->tick_offset, fc->predicted_time, fc->predicted_coherence,
               fc->predicted_decoherence, fc->confidence);
    }
    
    fclose(f);
    printf("[CHRONOMETRIC] Forecast report written\n");
}
