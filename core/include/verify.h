/**
 * @file verify.h
 * @brief System verification and checkpoint for Phase 6
 * 
 * Verifies system integrity before expansion:
 * - Coherence ≈ 0.999
 * - Decoherence < 0.001
 * - Ethics E ≥ 2.99
 */

#ifndef QALLOW_VERIFY_H
#define QALLOW_VERIFY_H

#include <stdint.h>

// Verification result codes
typedef enum {
    VERIFY_OK = 0,
    VERIFY_COHERENCE_LOW = 1,
    VERIFY_DECOHERENCE_HIGH = 2,
    VERIFY_ETHICS_LOW = 4,
    VERIFY_SANDBOX_FAIL = 8,
    VERIFY_TELEMETRY_FAIL = 16
} verify_result_t;

// Verification report
typedef struct {
    verify_result_t status;
    
    // Measurements
    double coherence;
    double decoherence;
    double ethics_score;
    double stability;
    
    // Thresholds
    double coherence_min;
    double decoherence_max;
    double ethics_min;
    
    // Flags
    int sandbox_active;
    int telemetry_active;
    int ethics_enforced;
    
    // Timestamp
    uint64_t timestamp;
    
    // Message
    char message[256];
} verify_report_t;

// Core API
int verify_system(verify_report_t* report);
void verify_print_report(const verify_report_t* report);
int verify_is_healthy(const verify_report_t* report);

#endif // QALLOW_VERIFY_H

