#ifndef CHRONOMETRIC_H
#define CHRONOMETRIC_H

#include "qallow_kernel.h"
#include <time.h>

// Chronometric Prediction Layer
// Temporal forecasting and delta-t tracking

#define CHRONO_HISTORY_SIZE 100
#define CHRONO_FORECAST_HORIZON 50

// Time Bank entry - tracks temporal prediction accuracy
typedef struct {
    double delta_t;           // Observed - Predicted time difference
    double confidence;        // Prediction confidence [0,1]
    double timestamp;         // When this entry was recorded
    int event_id;            // Event identifier
} chrono_bank_entry_t;

// Chronometric Time Bank
typedef struct {
    chrono_bank_entry_t history[CHRONO_HISTORY_SIZE];
    int history_count;
    int current_index;
    
    // Statistics
    double avg_delta_t;
    double std_delta_t;
    double overall_confidence;
    
    // Learning parameters
    double confidence_decay;
    double learning_rate;
    double adaptation_threshold;
} chrono_bank_t;

// Temporal forecast
typedef struct {
    int tick_offset;          // Ticks into the future
    double predicted_time;    // Predicted wall-clock time
    float predicted_coherence;
    float predicted_decoherence;
    float predicted_ethics;
    double confidence;
} temporal_forecast_t;

// Chronometric state
typedef struct {
    chrono_bank_t time_bank;
    temporal_forecast_t forecasts[CHRONO_FORECAST_HORIZON];
    int num_forecasts;
    
    // Timing tracking
    double simulation_start_time;
    double last_tick_time;
    double tick_duration_avg;
    double tick_duration_std;
    
    // Drift detection
    double accumulated_drift;
    double drift_rate;
    bool drift_alert_active;
    
    // Telemetry
    FILE* chrono_telemetry;
    char telemetry_file[128];
} chronometric_state_t;

// Function declarations

// Initialization
void chronometric_init(chronometric_state_t* chrono);
void chronometric_cleanup(chronometric_state_t* chrono);

// Time Bank operations
void chrono_bank_init(chrono_bank_t* bank);
void chrono_bank_record_event(chrono_bank_t* bank, 
                              int event_id,
                              double observed_time,
                              double predicted_time);
void chrono_bank_update_stats(chrono_bank_t* bank);
double chrono_bank_get_confidence(const chrono_bank_t* bank);

// Prediction
void chronometric_generate_forecasts(chronometric_state_t* chrono,
                                     const qallow_state_t* current_state,
                                     int horizon_ticks);
void chronometric_update_forecast(chronometric_state_t* chrono,
                                 const qallow_state_t* observed_state,
                                 int tick);

// Drift tracking
void chronometric_track_drift(chronometric_state_t* chrono,
                             double observed_time,
                             double expected_time);
void chronometric_update_tick_timing(chronometric_state_t* chrono,
                                    double tick_duration);
bool chronometric_detect_anomaly(const chronometric_state_t* chrono,
                                const qallow_state_t* state);

// Analysis
double chronometric_calculate_temporal_offset(const chronometric_state_t* chrono);
double chronometric_predict_next_tick_time(const chronometric_state_t* chrono);
void chronometric_analyze_patterns(chronometric_state_t* chrono);

// Telemetry
void chronometric_write_telemetry(chronometric_state_t* chrono, int tick);
void chronometric_write_forecast_report(const chronometric_state_t* chrono);

// Reporting
void chronometric_print_status(const chronometric_state_t* chrono);
void chronometric_print_forecasts(const chronometric_state_t* chrono);

// Utility functions
double get_wall_time(void);  // Returns current wall-clock time in seconds

#endif // CHRONOMETRIC_H