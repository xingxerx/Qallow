#ifndef TELEMETRY_H
#define TELEMETRY_H

#include <stdio.h>
#include <time.h>

typedef struct {
    FILE* stream_file;
    FILE* bench_file;
    int tick_count;
    double compile_ms;
    double run_ms;
    int mode;  // 0 = CPU, 1 = CUDA
} telemetry_t;

// Initialize telemetry system
void telemetry_init(telemetry_t* tel);

// Stream real-time tick data
void telemetry_stream_tick(telemetry_t* tel, double orbital, double river, double mycelial, 
                           double global, double decoherence, int mode);

// Log benchmark summary
void telemetry_log_benchmark(telemetry_t* tel, double compile_ms, double run_ms, 
                             double decoherence, double global, int mode);

// Flush and close files
void telemetry_close(telemetry_t* tel);

#endif

