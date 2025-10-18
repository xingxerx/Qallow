/**
 * @file ingest.c
 * @brief Data ingestion layer implementation
 * 
 * Handles incoming sensor/feed streams and converts them to normalized JSON packets
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ingest.h"

/**
 * Initialize ingestion manager
 */
void ingest_init(ingest_manager_t* mgr) {
    if (!mgr) return;
    
    memset(mgr, 0, sizeof(ingest_manager_t));
    mgr->buffer_head = 0;
    mgr->buffer_tail = 0;
    mgr->buffer_count = 0;
    mgr->running = 1;
    mgr->paused = 0;
    
    printf("[INGEST] Manager initialized\n");
}

/**
 * Cleanup ingestion manager
 */
void ingest_cleanup(ingest_manager_t* mgr) {
    if (!mgr) return;
    
    mgr->running = 0;
    printf("[INGEST] Manager cleaned up\n");
}

/**
 * Add a new data stream
 */
int ingest_add_stream(ingest_manager_t* mgr, const char* name, const char* endpoint) {
    if (!mgr || !name || !endpoint) return -1;
    if (mgr->stream_count >= INGEST_MAX_STREAMS) return -1;
    
    ingest_stream_t* stream = &mgr->streams[mgr->stream_count];
    strncpy(stream->name, name, sizeof(stream->name) - 1);
    strncpy(stream->endpoint, endpoint, sizeof(stream->endpoint) - 1);
    stream->enabled = 1;
    stream->packets_received = 0;
    stream->packets_dropped = 0;
    stream->last_value = 0.0;
    stream->last_update = time(NULL);
    
    mgr->stream_count++;
    printf("[INGEST] Stream added: %s -> %s\n", name, endpoint);
    return 0;
}

/**
 * Remove a data stream
 */
int ingest_remove_stream(ingest_manager_t* mgr, const char* name) {
    if (!mgr || !name) return -1;
    
    for (int i = 0; i < mgr->stream_count; i++) {
        if (strcmp(mgr->streams[i].name, name) == 0) {
            // Shift remaining streams
            for (int j = i; j < mgr->stream_count - 1; j++) {
                mgr->streams[j] = mgr->streams[j + 1];
            }
            mgr->stream_count--;
            printf("[INGEST] Stream removed: %s\n", name);
            return 0;
        }
    }
    return -1;
}

/**
 * Enable a stream
 */
int ingest_enable_stream(ingest_manager_t* mgr, const char* name) {
    if (!mgr || !name) return -1;
    
    for (int i = 0; i < mgr->stream_count; i++) {
        if (strcmp(mgr->streams[i].name, name) == 0) {
            mgr->streams[i].enabled = 1;
            printf("[INGEST] Stream enabled: %s\n", name);
            return 0;
        }
    }
    return -1;
}

/**
 * Disable a stream
 */
int ingest_disable_stream(ingest_manager_t* mgr, const char* name) {
    if (!mgr || !name) return -1;
    
    for (int i = 0; i < mgr->stream_count; i++) {
        if (strcmp(mgr->streams[i].name, name) == 0) {
            mgr->streams[i].enabled = 0;
            printf("[INGEST] Stream disabled: %s\n", name);
            return 0;
        }
    }
    return -1;
}

/**
 * Pause all streams
 */
int ingest_pause_all(ingest_manager_t* mgr) {
    if (!mgr) return -1;
    mgr->paused = 1;
    printf("[INGEST] All streams paused\n");
    return 0;
}

/**
 * Resume all streams
 */
int ingest_resume_all(ingest_manager_t* mgr) {
    if (!mgr) return -1;
    mgr->paused = 0;
    printf("[INGEST] All streams resumed\n");
    return 0;
}

/**
 * Push a packet into the buffer
 */
int ingest_push_packet(ingest_manager_t* mgr, const ingest_packet_t* packet) {
    if (!mgr || !packet) return -1;
    if (mgr->paused) return -1;
    
    if (mgr->buffer_count >= INGEST_BUFFER_SIZE) {
        mgr->total_dropped++;
        return -1; // Buffer full
    }
    
    mgr->packet_buffer[mgr->buffer_head] = *packet;
    mgr->buffer_head = (mgr->buffer_head + 1) % INGEST_BUFFER_SIZE;
    mgr->buffer_count++;
    mgr->total_packets++;
    
    return 0;
}

/**
 * Pop a packet from the buffer
 */
int ingest_pop_packet(ingest_manager_t* mgr, ingest_packet_t* packet) {
    if (!mgr || !packet) return -1;
    if (mgr->buffer_count == 0) return -1;
    
    *packet = mgr->packet_buffer[mgr->buffer_tail];
    mgr->buffer_tail = (mgr->buffer_tail + 1) % INGEST_BUFFER_SIZE;
    mgr->buffer_count--;
    
    return 0;
}

/**
 * Peek at the next packet without removing it
 */
int ingest_peek_packet(ingest_manager_t* mgr, ingest_packet_t* packet) {
    if (!mgr || !packet) return -1;
    if (mgr->buffer_count == 0) return -1;
    
    *packet = mgr->packet_buffer[mgr->buffer_tail];
    return 0;
}

/**
 * Get packet count
 */
int ingest_packet_count(ingest_manager_t* mgr) {
    if (!mgr) return 0;
    return mgr->buffer_count;
}

/**
 * Print ingestion statistics
 */
void ingest_print_stats(ingest_manager_t* mgr) {
    if (!mgr) return;
    
    printf("\n╔════════════════════════════════════════╗\n");
    printf("║     INGESTION STATISTICS               ║\n");
    printf("╚════════════════════════════════════════╝\n\n");
    printf("Total Packets: %llu\n", (unsigned long long)mgr->total_packets);
    printf("Dropped Packets: %llu\n", (unsigned long long)mgr->total_dropped);
    printf("Buffer Usage: %d / %d\n", mgr->buffer_count, INGEST_BUFFER_SIZE);
    printf("Status: %s\n", mgr->paused ? "PAUSED" : "RUNNING");
}

/**
 * Print active streams
 */
void ingest_print_streams(ingest_manager_t* mgr) {
    if (!mgr) return;
    
    printf("\n╔════════════════════════════════════════╗\n");
    printf("║     ACTIVE STREAMS                     ║\n");
    printf("╚════════════════════════════════════════╝\n\n");
    
    for (int i = 0; i < mgr->stream_count; i++) {
        ingest_stream_t* stream = &mgr->streams[i];
        printf("[%d] %s\n", i, stream->name);
        printf("    Endpoint: %s\n", stream->endpoint);
        printf("    Status: %s\n", stream->enabled ? "ENABLED" : "DISABLED");
        printf("    Packets: %llu (Dropped: %llu)\n", 
               (unsigned long long)stream->packets_received,
               (unsigned long long)stream->packets_dropped);
        printf("    Last Value: %.6f\n\n", stream->last_value);
    }
}

