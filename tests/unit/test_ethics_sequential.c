/**
 * test_ethics_sequential.c
 * Unit tests for sequential ethics decision logging (Phase 8-10 enhancement)
 * Tests the step-by-step audit trail for ethics decisions
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ethics_core.h"

#define TEST_LOG_PATH "/tmp/ethics_trace_test.csv"

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

static void test_sequential_step_logging(void) {
    printf("Test: Sequential step logging...\n");
    cleanup_test_log();
    
    ethics_sequential_step_t step = {
        .step_id = 0,
        .timestamp_ms = 1000,
        .rule_name = "safety_check",
        .input_value = 0.85,
        .threshold = 0.70,
        .verdict = 1,
        .intervention_type = "none"
    };
    
    int rc = ethics_log_sequential_step(&step, TEST_LOG_PATH);
    assert(rc == 0);
    
    // Verify file was created and has header + 1 data line
    int lines = count_csv_lines(TEST_LOG_PATH);
    assert(lines == 2);  // header + 1 data line
    
    printf("  ✓ Sequential step logged successfully\n");
    cleanup_test_log();
}

static void test_decision_sequence_trace(void) {
    printf("Test: Decision sequence trace...\n");
    cleanup_test_log();
    
    ethics_model_t model;
    ethics_model_default(&model);
    
    ethics_metrics_t metrics = {
        .safety = 0.85,
        .clarity = 0.80,
        .human = 0.75,
        .reality_drift = 0.10
    };
    
    int rc = ethics_trace_decision_sequence(&model, &metrics, TEST_LOG_PATH);
    assert(rc == 0);
    
    // Verify all 5 steps were logged (header + 5 steps)
    int lines = count_csv_lines(TEST_LOG_PATH);
    assert(lines == 6);  // header + 5 decision steps
    
    printf("  ✓ Decision sequence traced with 5 steps\n");
    cleanup_test_log();
}

static void test_sequential_consistency(void) {
    printf("Test: Sequential verdict consistency...\n");
    cleanup_test_log();
    
    ethics_model_t model;
    ethics_model_default(&model);
    
    // Test case 1: All metrics pass
    ethics_metrics_t metrics_pass = {
        .safety = 0.85,
        .clarity = 0.80,
        .human = 0.75,
        .reality_drift = 0.10
    };
    
    ethics_trace_decision_sequence(&model, &metrics_pass, TEST_LOG_PATH);
    int pass_lines = count_csv_lines(TEST_LOG_PATH);
    assert(pass_lines == 6);
    
    // Test case 2: Some metrics fail
    cleanup_test_log();
    ethics_metrics_t metrics_fail = {
        .safety = 0.50,  // Below threshold
        .clarity = 0.80,
        .human = 0.75,
        .reality_drift = 0.10
    };
    
    ethics_trace_decision_sequence(&model, &metrics_fail, TEST_LOG_PATH);
    int fail_lines = count_csv_lines(TEST_LOG_PATH);
    assert(fail_lines == 6);
    
    printf("  ✓ Sequential verdicts consistent across test cases\n");
    cleanup_test_log();
}

static void test_multiple_sequential_traces(void) {
    printf("Test: Multiple sequential traces (audit trail)...\n");
    cleanup_test_log();
    
    ethics_model_t model;
    ethics_model_default(&model);
    
    ethics_metrics_t metrics = {
        .safety = 0.85,
        .clarity = 0.80,
        .human = 0.75,
        .reality_drift = 0.10
    };
    
    // Simulate multiple decision sequences
    for (int i = 0; i < 3; i++) {
        ethics_trace_decision_sequence(&model, &metrics, TEST_LOG_PATH);
    }
    
    // Verify all traces were appended (header + 3*5 steps)
    int lines = count_csv_lines(TEST_LOG_PATH);
    assert(lines == 16);  // header + 15 steps (3 traces × 5 steps)
    
    printf("  ✓ Multiple traces appended to audit trail\n");
    cleanup_test_log();
}

static void test_intervention_logging(void) {
    printf("Test: Intervention type logging...\n");
    cleanup_test_log();
    
    ethics_model_t model;
    ethics_model_default(&model);
    
    // Test with failing metrics to trigger interventions
    ethics_metrics_t metrics = {
        .safety = 0.50,      // Below threshold -> safety_intervention
        .clarity = 0.50,     // Below threshold -> clarity_intervention
        .human = 0.50,       // Below threshold -> human_intervention
        .reality_drift = 0.50  // Above threshold -> reality_correction
    };
    
    int rc = ethics_trace_decision_sequence(&model, &metrics, TEST_LOG_PATH);
    assert(rc == 0);
    
    // Verify interventions were logged
    FILE* f = fopen(TEST_LOG_PATH, "r");
    assert(f != NULL);
    
    char buf[512];
    int intervention_count = 0;
    while (fgets(buf, sizeof(buf), f)) {
        if (strstr(buf, "intervention") || strstr(buf, "correction")) {
            intervention_count++;
        }
    }
    fclose(f);
    
    // Should have at least 4 interventions (safety, clarity, human, reality)
    assert(intervention_count >= 4);
    
    printf("  ✓ Interventions logged correctly\n");
    cleanup_test_log();
}

int main(void) {
    printf("\n");
    printf("════════════════════════════════════════════════════════════════\n");
    printf("  Sequential Ethics Decision Logging Tests (Phase 8-10)\n");
    printf("════════════════════════════════════════════════════════════════\n\n");
    
    test_sequential_step_logging();
    test_decision_sequence_trace();
    test_sequential_consistency();
    test_multiple_sequential_traces();
    test_intervention_logging();
    
    printf("\n");
    printf("════════════════════════════════════════════════════════════════\n");
    printf("  ✓ All sequential ethics tests passed!\n");
    printf("════════════════════════════════════════════════════════════════\n\n");
    
    return 0;
}

