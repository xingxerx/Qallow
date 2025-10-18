/**
 * @file ingest.h
 * @brief Data ingestion layer for Phase 6 - Live Interface and External Data Integration
 * 
 * Handles incoming sensor/feed streams and converts them to normalized JSON packets
 * for processing through the Qallow VM pipeline.
 */

#ifndef QALLOW_INGEST_H
#define QALLOW_INGEST_H

#include <time.h>
#include <stdint.h>

// Maximum packet size for ingested data
#define INGEST_MAX_PACKET_SIZE 4096
#define INGEST_MAX_STREAMS 16
#define INGEST_BUFFER_SIZE 65536

// Data packet types
typedef enum {
    INGEST_TYPE_TELEMETRY = 0,
    INGEST_TYPE_SENSOR = 1,
    INGEST_TYPE_CONTROL = 2,
    INGEST_TYPE_FEEDBACK = 3,
    INGEST_TYPE_OVERRIDE = 4
} ingest_packet_type_t;

// Normalized data packet structure
typedef struct {
    uint64_t timestamp;           // Unix timestamp
    ingest_packet_type_t type;    // Packet type
    double value;                 // Primary value
    double confidence;            // Confidence score (0.0-1.0)
    char source[64];              // Source identifier
    char metadata[256];           // Additional metadata
} ingest_packet_t;

// Stream adapter interface
typedef struct {
    char name[64];                // Adapter name
    char endpoint[256];           // Connection endpoint (URL, serial port, etc.)
    int enabled;                  // Is this stream active?
    uint64_t packets_received;    // Total packets received
    uint64_t packets_dropped;     // Dropped packets
    double last_value;            // Last received value
    time_t last_update;           // Last update timestamp
} ingest_stream_t;

// Ingestion manager
typedef struct {
    ingest_stream_t streams[INGEST_MAX_STREAMS];
    int stream_count;
    
    // Circular buffer for incoming packets
    ingest_packet_t packet_buffer[INGEST_BUFFER_SIZE];
    int buffer_head;
    int buffer_tail;
    int buffer_count;
    
    // Statistics
    uint64_t total_packets;
    uint64_t total_dropped;
    double avg_latency_ms;
    
    // Control flags
    int paused;
    int running;
} ingest_manager_t;

// Core API
void ingest_init(ingest_manager_t* mgr);
void ingest_cleanup(ingest_manager_t* mgr);

// Stream management
int ingest_add_stream(ingest_manager_t* mgr, const char* name, const char* endpoint);
int ingest_remove_stream(ingest_manager_t* mgr, const char* name);
int ingest_enable_stream(ingest_manager_t* mgr, const char* name);
int ingest_disable_stream(ingest_manager_t* mgr, const char* name);
int ingest_pause_all(ingest_manager_t* mgr);
int ingest_resume_all(ingest_manager_t* mgr);

// Packet operations
int ingest_push_packet(ingest_manager_t* mgr, const ingest_packet_t* packet);
int ingest_pop_packet(ingest_manager_t* mgr, ingest_packet_t* packet);
int ingest_peek_packet(ingest_manager_t* mgr, ingest_packet_t* packet);
int ingest_packet_count(ingest_manager_t* mgr);

// Statistics and monitoring
void ingest_print_stats(ingest_manager_t* mgr);
void ingest_print_streams(ingest_manager_t* mgr);

#endif // QALLOW_INGEST_H

