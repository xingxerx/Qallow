#include "telemetry.h"
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

void telemetry_init(telemetry_t* tel) {
    if (!tel) return;
    
    mkdir("data", 0755);
    mkdir("data/logs", 0755);

    // Open streaming CSV file
    tel->stream_file = fopen("data/logs/telemetry_stream.csv", "w");
    if (tel->stream_file) {
        fprintf(tel->stream_file, "tick,orbital,river,mycelial,global,deco,mode\n");
        fflush(tel->stream_file);
    }
    
    // Open benchmark log file
    tel->bench_file = fopen("data/logs/qallow_bench.log", "a");
    if (tel->bench_file) {
        fprintf(tel->bench_file, "timestamp,compile_ms,run_ms,deco,global,mode\n");
        fflush(tel->bench_file);
    }
    
    tel->tick_count = 0;
    tel->compile_ms = 0.0;
    tel->run_ms = 0.0;
    tel->mode = 0;
    
    printf("[TELEMETRY] System initialized\n");
    printf("[TELEMETRY] Streaming to: data/logs/telemetry_stream.csv\n");
    printf("[TELEMETRY] Logging to: data/logs/qallow_bench.log\n");
}

void telemetry_stream_tick(telemetry_t* tel, double orbital, double river, double mycelial,
                           double global, double decoherence, int mode) {
    if (!tel || !tel->stream_file) return;
    
    fprintf(tel->stream_file, "%d,%.4f,%.4f,%.4f,%.4f,%.5f,%s\n",
            tel->tick_count,
            orbital, river, mycelial, global, decoherence,
            mode == 1 ? "CUDA" : "CPU");
    
    // Flush every 10 ticks for real-time visibility
    if (tel->tick_count % 10 == 0) {
        fflush(tel->stream_file);
    }
    
    tel->tick_count++;
}

void telemetry_log_benchmark(telemetry_t* tel, double compile_ms, double run_ms,
                             double decoherence, double global, int mode) {
    if (!tel || !tel->bench_file) return;
    
    time_t now = time(NULL);
    struct tm* timeinfo = localtime(&now);
    char timestamp[32];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", timeinfo);
    
    fprintf(tel->bench_file, "%s,%.1f,%.2f,%.5f,%.4f,%s\n",
            timestamp, compile_ms, run_ms, decoherence, global,
            mode == 1 ? "CUDA" : "CPU");
    
    fflush(tel->bench_file);
    
    printf("[TELEMETRY] Benchmark logged: compile=%.1fms, run=%.2fms, mode=%s\n",
           compile_ms, run_ms, mode == 1 ? "CUDA" : "CPU");
}

void telemetry_close(telemetry_t* tel) {
    if (!tel) return;
    
    if (tel->stream_file) {
        fflush(tel->stream_file);
        fclose(tel->stream_file);
        tel->stream_file = NULL;
    }
    
    if (tel->bench_file) {
        fflush(tel->bench_file);
        fclose(tel->bench_file);
        tel->bench_file = NULL;
    }
    
    printf("[TELEMETRY] System closed\n");
}

