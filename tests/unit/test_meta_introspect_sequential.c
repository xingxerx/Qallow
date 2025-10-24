/**
 * test_meta_introspect_sequential.c
 * Unit tests for sequential meta-introspection (Phase 16 stabilization)
 * Tests the step-by-step reasoning for introspection triggers
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "meta_introspect.h"

#define TEST_LOG_PATH "/tmp/introspect_trace_test.csv"

static void cleanup_test_log(void) {
    unlink(TEST_LOG_PATH);
}

static int count_csv_lines(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) return 0;
    
    int count = 0;
    char buf[512];
    while (fgets(buf, sizeof(buf), f)) {
        count++;
    }
    fclose(f);
    return count;
}

static void test_trigger_logging(void) {
    printf("Test: Introspection trigger logging...\n");
    cleanup_test_log();
    
    introspection_trigger_t trigger = {
        .trigger_id = 1,
        .timestamp_ms = 1000,
        .trigger_type = "coherence_drop",
        .metric_value = 0.65,
        .threshold = 0.80,
        .severity = 1
    };
    
    int rc = meta_introspect_log_trigger(&trigger, TEST_LOG_PATH);
    assert(rc == 0);
    
    // Verify file was created and has header + 1 data line
    int lines = count_csv_lines(TEST_LOG_PATH);
    assert(lines == 2);  // header + 1 trigger
    
    printf("  ✓ Trigger logged successfully\n");
    cleanup_test_log();
}

static void test_sequential_reasoning_coherence(void) {
    printf("Test: Sequential reasoning for coherence drop...\n");
    cleanup_test_log();
    
    introspection_trigger_t trigger = {
        .trigger_id = 1,
        .timestamp_ms = 1000,
        .trigger_type = "coherence_drop",
        .metric_value = 0.65,
        .threshold = 0.80,
        .severity = 1
    };
    
    introspection_result_t result;
    int rc = meta_introspect_sequential_reasoning(&trigger, &result, TEST_LOG_PATH);
    assert(rc == 0);
    
    // Verify result
    assert(result.trigger_id == 1);
    assert(result.introspection_score > 0.0f && result.introspection_score <= 1.0f);
    assert(result.confidence > 0);
    assert(result.recommendation != NULL);
    
    printf("  ✓ Coherence reasoning: score=%.3f, confidence=%d, rec=%s\n",
           result.introspection_score, result.confidence, result.recommendation);
    
    cleanup_test_log();
}

static void test_sequential_reasoning_ethics(void) {
    printf("Test: Sequential reasoning for ethics violation...\n");
    cleanup_test_log();
    
    introspection_trigger_t trigger = {
        .trigger_id = 2,
        .timestamp_ms = 2000,
        .trigger_type = "ethics_violation",
        .metric_value = 0.45,
        .threshold = 0.70,
        .severity = 2  // high severity
    };
    
    introspection_result_t result;
    int rc = meta_introspect_sequential_reasoning(&trigger, &result, TEST_LOG_PATH);
    assert(rc == 0);
    
    // Verify result
    assert(result.trigger_id == 2);
    assert(result.introspection_score < 0.5f);  // Should be low for ethics violation
    assert(result.confidence >= 80);  // High confidence for ethics
    assert(strcmp(result.recommendation, "apply_ethics_intervention") == 0);
    
    printf("  ✓ Ethics reasoning: score=%.3f, confidence=%d, rec=%s\n",
           result.introspection_score, result.confidence, result.recommendation);
    
    cleanup_test_log();
}

static void test_sequential_reasoning_latency(void) {
    printf("Test: Sequential reasoning for latency spike...\n");
    cleanup_test_log();
    
    introspection_trigger_t trigger = {
        .trigger_id = 3,
        .timestamp_ms = 3000,
        .trigger_type = "latency_spike",
        .metric_value = 250.0f,
        .threshold = 100.0f,
        .severity = 1
    };
    
    introspection_result_t result;
    int rc = meta_introspect_sequential_reasoning(&trigger, &result, TEST_LOG_PATH);
    assert(rc == 0);
    
    // Verify result
    assert(result.trigger_id == 3);
    assert(result.introspection_score >= 0.0f && result.introspection_score <= 1.0f);
    assert(result.confidence > 0);
    
    printf("  ✓ Latency reasoning: score=%.3f, confidence=%d, rec=%s\n",
           result.introspection_score, result.confidence, result.recommendation);
    
    cleanup_test_log();
}

static void test_severity_adjustment(void) {
    printf("Test: Severity-based score adjustment...\n");
    cleanup_test_log();
    
    // Test low severity
    introspection_trigger_t trigger_low = {
        .trigger_id = 1,
        .timestamp_ms = 1000,
        .trigger_type = "coherence_drop",
        .metric_value = 0.65,
        .threshold = 0.80,
        .severity = 0  // low
    };
    
    introspection_result_t result_low;
    meta_introspect_sequential_reasoning(&trigger_low, &result_low, TEST_LOG_PATH);
    
    // Test high severity (same metric)
    introspection_trigger_t trigger_high = {
        .trigger_id = 2,
        .timestamp_ms = 2000,
        .trigger_type = "coherence_drop",
        .metric_value = 0.65,
        .threshold = 0.80,
        .severity = 2  // high
    };
    
    introspection_result_t result_high;
    meta_introspect_sequential_reasoning(&trigger_high, &result_high, TEST_LOG_PATH);
    
    // High severity should have lower score
    assert(result_high.introspection_score < result_low.introspection_score);
    
    printf("  ✓ Severity adjustment: low=%.3f, high=%.3f\n",
           result_low.introspection_score, result_high.introspection_score);
    
    cleanup_test_log();
}

static void test_multiple_triggers_audit_trail(void) {
    printf("Test: Multiple triggers audit trail...\n");
    cleanup_test_log();
    
    // Simulate multiple triggers
    for (int i = 0; i < 5; i++) {
        introspection_trigger_t trigger = {
            .trigger_id = i,
            .timestamp_ms = 1000 + (i * 100),
            .trigger_type = i % 3 == 0 ? "coherence_drop" : 
                           i % 3 == 1 ? "ethics_violation" : "latency_spike",
            .metric_value = 0.5f + (i * 0.1f),
            .threshold = 0.80f,
            .severity = i % 3
        };
        
        introspection_result_t result;
        meta_introspect_sequential_reasoning(&trigger, &result, TEST_LOG_PATH);
    }
    
    // Verify all triggers were logged
    int lines = count_csv_lines(TEST_LOG_PATH);
    assert(lines >= 6);  // header + 5 triggers + results
    
    printf("  ✓ Multiple triggers logged to audit trail\n");
    cleanup_test_log();
}

int main(void) {
    printf("\n");
    printf("════════════════════════════════════════════════════════════════\n");
    printf("  Sequential Meta-Introspection Tests (Phase 16)\n");
    printf("════════════════════════════════════════════════════════════════\n\n");
    
    test_trigger_logging();
    test_sequential_reasoning_coherence();
    test_sequential_reasoning_ethics();
    test_sequential_reasoning_latency();
    test_severity_adjustment();
    test_multiple_triggers_audit_trail();
    
    printf("\n");
    printf("════════════════════════════════════════════════════════════════\n");
    printf("  ✓ All sequential introspection tests passed!\n");
    printf("════════════════════════════════════════════════════════════════\n\n");
    
    return 0;
}

