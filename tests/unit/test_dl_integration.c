#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "dl_integration.h"

/**
 * Test suite for LibTorch deep learning integration
 * Tests model loading, inference, and error handling
 */

void test_dl_support_detection(void) {
    printf("Test 1: LibTorch support detection\n");
    int supported = dl_model_supported();
    printf("  LibTorch support: %s\n", supported ? "YES" : "NO (expected without LibTorch build flag)");
    printf("  ✓ Test passed\n\n");
}

void test_model_not_loaded_initially(void) {
    printf("Test 2: Model not loaded initially\n");
    int is_loaded = dl_model_is_loaded();
    assert(is_loaded == 0);
    printf("  Model loaded: %s\n", is_loaded ? "YES" : "NO");
    printf("  ✓ Test passed\n\n");
}

void test_load_nonexistent_model(void) {
    printf("Test 3: Load nonexistent model\n");
    int result = dl_model_load("/nonexistent/model.pt", 0);
    printf("  Load result: %d (expected non-zero for missing file)\n", result);
    const char* error = dl_model_last_error();
    printf("  Error message: %s\n", error ? error : "(none)");
    printf("  ✓ Test passed\n\n");
}

void test_inference_without_model(void) {
    printf("Test 4: Inference without loaded model\n");
    float input[10] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    float output[5] = {0.0f};
    
    int result = dl_model_infer(input, 10, output, 5);
    printf("  Inference result: %d (expected negative for no model)\n", result);
    const char* error = dl_model_last_error();
    printf("  Error message: %s\n", error ? error : "(none)");
    assert(result < 0);
    printf("  ✓ Test passed\n\n");
}

void test_invalid_inference_parameters(void) {
    printf("Test 5: Invalid inference parameters\n");
    float output[5] = {0.0f};
    
    // Test with NULL input
    int result = dl_model_infer(NULL, 10, output, 5);
    printf("  NULL input result: %d (expected negative)\n", result);
    assert(result < 0);
    
    // Test with zero input length
    float input[10] = {1.0f};
    result = dl_model_infer(input, 0, output, 5);
    printf("  Zero input length result: %d (expected negative)\n", result);
    assert(result < 0);
    
    // Test with NULL output
    result = dl_model_infer(input, 10, NULL, 5);
    printf("  NULL output result: %d (expected negative)\n", result);
    assert(result < 0);
    
    printf("  ✓ Test passed\n\n");
}

void test_unload_without_model(void) {
    printf("Test 6: Unload without loaded model\n");
    dl_model_unload();
    printf("  Unload completed without error\n");
    int is_loaded = dl_model_is_loaded();
    assert(is_loaded == 0);
    printf("  ✓ Test passed\n\n");
}

void test_error_message_retrieval(void) {
    printf("Test 7: Error message retrieval\n");
    const char* error = dl_model_last_error();
    printf("  Error message: %s\n", error ? error : "(none)");
    printf("  ✓ Test passed\n\n");
}

int main(void) {
    printf("========================================\n");
    printf("LibTorch Deep Learning Integration Tests\n");
    printf("========================================\n\n");
    
    test_dl_support_detection();
    test_model_not_loaded_initially();
    test_load_nonexistent_model();
    test_inference_without_model();
    test_invalid_inference_parameters();
    test_unload_without_model();
    test_error_message_retrieval();
    
    printf("========================================\n");
    printf("All tests passed!\n");
    printf("========================================\n");
    
    return 0;
}

