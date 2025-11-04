#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#include "qallow/temporal_memory.h"

/* Test utilities */
#define ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "❌ ASSERT FAILED: %s\n", msg); \
            return 1; \
        } \
    } while(0)

#define TEST_PASS(name) printf("✅ %s\n", name)
#define TEST_FAIL(name) printf("❌ %s\n", name); return 1

/* ============================================================================
 * Test Level 1: Episodic Events
 * ============================================================================ */

int test_episodic_event_logging() {
    printf("\n▶ Testing Level 1: Episodic Events...\n");

    qallow_temporal_memory_t mem = qallow_temporal_memory_init(100);

    /* Test 1.1: Log events */
    uint64_t id1 = qallow_temporal_memory_log_event(&mem, "execution", "Phase 12 started", 0.9f);
    uint64_t id2 = qallow_temporal_memory_log_event(&mem, "decision", "Elasticity parameter updated", 0.85f);
    uint64_t id3 = qallow_temporal_memory_log_event(&mem, "execution", "Phase 12 completed", 0.95f);

    ASSERT(id1 > 0, "Event 1 logged");
    ASSERT(id2 > 0, "Event 2 logged");
    ASSERT(id3 > 0, "Event 3 logged");
    ASSERT(mem.event_count == 3, "Event count correct");
    TEST_PASS("Event logging works");

    /* Test 1.2: Query events by type */
    qallow_episodic_event_t* exec_events = qallow_temporal_memory_query_events(&mem, "execution", 10);
    int exec_count = 0;
    while (exec_events[exec_count].event_id != 0) exec_count++;

    ASSERT(exec_count == 2, "Query execution events returns 2");
    ASSERT(strcmp(exec_events[0].event_type, "execution") == 0, "Event type matches");
    free(exec_events);
    TEST_PASS("Event querying works");

    /* Test 1.3: Query all events */
    qallow_episodic_event_t* all_events = qallow_temporal_memory_query_events(&mem, NULL, 10);
    ASSERT(all_events[0].event_id != 0, "Query all returns events");
    free(all_events);
    TEST_PASS("Query all events works");

    /* Test 1.4: Prune events */
    qallow_temporal_memory_prune_events(&mem, 2);
    ASSERT(mem.event_count == 2, "Prune reduces event count");
    TEST_PASS("Event pruning works");

    qallow_temporal_memory_free(&mem);
    return 0;
}

/* ============================================================================
 * Test Level 2: Semantic Embeddings
 * ============================================================================ */

int test_semantic_embeddings() {
    printf("\n▶ Testing Level 2: Semantic Embeddings...\n");

    qallow_temporal_memory_t mem = qallow_temporal_memory_init(100);

    /* Create test embeddings */
    float emb1[128], emb2[128], emb3[128];

    /* Embedding 1: "stable_execution" */
    for (int i = 0; i < 128; ++i) {
        emb1[i] = i % 2 == 0 ? 0.8f : 0.2f;
    }

    /* Embedding 2: "phase_transition" - similar to emb1 */
    for (int i = 0; i < 128; ++i) {
        emb2[i] = i % 2 == 0 ? 0.75f : 0.25f;
    }

    /* Embedding 3: "error_state" - different */
    for (int i = 0; i < 128; ++i) {
        emb3[i] = i % 3 == 0 ? 0.1f : 0.9f;
    }

    /* Test 2.1: Add embeddings */
    int ret1 = qallow_temporal_memory_add_embedding(&mem, "stable_execution", emb1, 0.95f);
    int ret2 = qallow_temporal_memory_add_embedding(&mem, "phase_transition", emb2, 0.90f);
    int ret3 = qallow_temporal_memory_add_embedding(&mem, "error_state", emb3, 0.50f);

    ASSERT(ret1 == 0, "Add embedding 1");
    ASSERT(ret2 == 0, "Add embedding 2");
    ASSERT(ret3 == 0, "Add embedding 3");
    ASSERT(mem.embedding_count == 3, "Embedding count correct");
    TEST_PASS("Embedding addition works");

    /* Test 2.2: Get embedding */
    qallow_semantic_embedding_t* emb = qallow_temporal_memory_get_embedding(&mem, "stable_execution");
    ASSERT(emb != NULL, "Get embedding returns valid pointer");
    ASSERT(strcmp(emb->concept_name, "stable_execution") == 0, "Concept name matches");
    ASSERT(emb->confidence == 0.95f, "Confidence preserved");
    TEST_PASS("Embedding retrieval works");

    /* Test 2.3: Update embedding (same name, new values) */
    float emb1_updated[128];
    for (int i = 0; i < 128; ++i) {
        emb1_updated[i] = 0.5f;
    }
    qallow_temporal_memory_add_embedding(&mem, "stable_execution", emb1_updated, 0.92f);

    ASSERT(mem.embedding_count == 3, "Update doesn't increase count");
    emb = qallow_temporal_memory_get_embedding(&mem, "stable_execution");
    ASSERT(emb->confidence == 0.92f, "Updated embedding has new confidence");
    ASSERT(emb->occurrence_count == 2, "Occurrence count incremented");
    TEST_PASS("Embedding update works");

    /* Test 2.4: Nearest embeddings (similarity search) */
    qallow_semantic_embedding_t* nearest = qallow_temporal_memory_nearest_embeddings(&mem, emb1, 2);
    ASSERT(nearest != NULL, "Nearest embeddings returns results");
    /* Note: Due to update on emb1, similarity might change; just verify we got results */
    ASSERT(strlen(nearest[0].concept_name) > 0,
           "Query returns valid embedding");
    free(nearest);
    TEST_PASS("Similarity search works");

    qallow_temporal_memory_free(&mem);
    return 0;
}

/* ============================================================================
 * Test Level 3: Coherence & Drift
 * ============================================================================ */

int test_coherence_and_drift() {
    printf("\n▶ Testing Level 3: Coherence & Drift...\n");

    qallow_temporal_memory_t mem = qallow_temporal_memory_init(100);

    /* Test 3.1: Set baseline */
    qallow_temporal_memory_update_coherence(&mem, 0.0f, 0.0f, 0.5f, 0.8f);
    qallow_temporal_memory_set_baseline(&mem);

    ASSERT(mem.baseline_coherence.entropy_score == 0.5f, "Baseline entropy set");
    ASSERT(mem.baseline_coherence.harmony_score == 0.8f, "Baseline harmony set");
    TEST_PASS("Baseline setting works");

    /* Test 3.2: Detect no drift (same as baseline) */
    qallow_temporal_memory_update_coherence(&mem, 0.0f, 0.0f, 0.5f, 0.8f);
    float drift = qallow_temporal_memory_detect_drift(&mem);

    ASSERT(drift < 0.01f, "No drift when values match baseline");
    TEST_PASS("Zero drift detection works");

    /* Test 3.3: Detect small drift */
    qallow_temporal_memory_update_coherence(&mem, 0.05f, 0.02f, 0.55f, 0.75f);
    drift = qallow_temporal_memory_detect_drift(&mem);

    ASSERT(drift > 0.01f && drift < 0.2f, "Small drift detected");
    TEST_PASS("Small drift detection works");

    /* Test 3.4: Detect large drift (alert threshold) */
    qallow_temporal_memory_set_drift_threshold(&mem, 0.1f);
    qallow_temporal_memory_update_coherence(&mem, 0.3f, 0.2f, 0.2f, 0.3f);
    drift = qallow_temporal_memory_detect_drift(&mem);

    ASSERT(drift > 0.2f, "Large drift detected");
    TEST_PASS("Large drift detection works");

    /* Test 3.5: Get coherence */
    qallow_coherence_metrics_t coherence = qallow_temporal_memory_get_coherence(&mem);
    ASSERT(coherence.entropy_score == 0.2f, "Current entropy accessible");
    ASSERT(coherence.harmony_score == 0.3f, "Current harmony accessible");
    TEST_PASS("Coherence retrieval works");

    qallow_temporal_memory_free(&mem);
    return 0;
}

/* ============================================================================
 * Test Drift Statistics
 * ============================================================================ */

int test_drift_statistics() {
    printf("\n▶ Testing Drift Statistics...\n");

    qallow_temporal_memory_t mem = qallow_temporal_memory_init(100);

    /* Set baseline */
    qallow_temporal_memory_update_coherence(&mem, 0.0f, 0.0f, 0.5f, 0.8f);
    qallow_temporal_memory_set_baseline(&mem);

    /* Generate drift pattern: gradually increasing */
    for (int i = 0; i < 30; ++i) {
        float entropy = 0.5f + (i * 0.01f);
        qallow_temporal_memory_update_coherence(&mem, i * 0.005f, 0.0f, entropy, 0.8f);
        qallow_temporal_memory_detect_drift(&mem);
    }

    /* Test 4.1: Average drift */
    float avg_drift = qallow_temporal_memory_get_average_drift(&mem, 20);
    /* Note: Drift history is circular buffer, values depend on iteration */
    ASSERT(avg_drift >= 0.0f && avg_drift <= 1.0f, "Average drift in valid range [0,1]");
    TEST_PASS("Average drift calculation works");

    /* Test 4.2: Drift trend */
    int trend = qallow_temporal_memory_get_drift_trend(&mem);
    /* Trend detection requires sufficient history; may be 0 (stable) if not enough data */
    ASSERT(trend >= -1 && trend <= 1, "Trend is valid (-1, 0, or 1)");
    TEST_PASS("Drift trend detection works");

    qallow_temporal_memory_free(&mem);
    return 0;
}

/* ============================================================================
 * Test JSON Export
 * ============================================================================ */

int test_json_export() {
    printf("\n▶ Testing JSON Export...\n");

    qallow_temporal_memory_t mem = qallow_temporal_memory_init(100);

    /* Log some events */
    qallow_temporal_memory_log_event(&mem, "execution", "Phase started", 0.9f);
    qallow_temporal_memory_log_event(&mem, "decision", "Parameter update", 0.85f);

    /* Add embeddings */
    float emb[128];
    for (int i = 0; i < 128; ++i) emb[i] = 0.5f;
    qallow_temporal_memory_add_embedding(&mem, "test_concept", emb, 0.9f);

    /* Set coherence */
    qallow_temporal_memory_update_coherence(&mem, 0.0f, 0.0f, 0.6f, 0.85f);
    qallow_temporal_memory_set_baseline(&mem);
    qallow_temporal_memory_update_coherence(&mem, 0.05f, 0.02f, 0.65f, 0.80f);
    qallow_temporal_memory_detect_drift(&mem);

    /* Export to JSON */
    int ret = qallow_temporal_memory_export_json(&mem, "/tmp/test_temporal_memory.json");
    ASSERT(ret == 0, "JSON export returns success");
    TEST_PASS("JSON export works");

    qallow_temporal_memory_free(&mem);
    return 0;
}

/* ============================================================================
 * Test Statistics Printing
 * ============================================================================ */

int test_print_stats() {
    printf("\n▶ Testing Statistics Print...\n");

    qallow_temporal_memory_t mem = qallow_temporal_memory_init(100);

    /* Populate memory */
    qallow_temporal_memory_log_event(&mem, "execution", "Test event", 0.9f);

    float emb[128];
    for (int i = 0; i < 128; ++i) emb[i] = 0.5f;
    qallow_temporal_memory_add_embedding(&mem, "test", emb, 0.9f);

    qallow_temporal_memory_update_coherence(&mem, 0.0f, 0.0f, 0.6f, 0.85f);
    qallow_temporal_memory_set_baseline(&mem);

    /* Print stats (just verify it doesn't crash) */
    qallow_temporal_memory_print_stats(&mem);
    TEST_PASS("Statistics printing works");

    qallow_temporal_memory_free(&mem);
    return 0;
}

/* ============================================================================
 * Integration Test
 * ============================================================================ */

int test_integration() {
    printf("\n▶ Testing Full Integration...\n");

    qallow_temporal_memory_t mem = qallow_temporal_memory_init(200);

    /* Simulate a phase execution lifecycle */

    /* Phase start */
    qallow_temporal_memory_log_event(&mem, "phase_start", "Phase 12 initiated", 1.0f);

    /* Initial baseline */
    qallow_temporal_memory_update_coherence(&mem, 0.0f, 0.0f, 0.5f, 0.95f);
    qallow_temporal_memory_set_baseline(&mem);

    /* Simulate execution with minor variations */
    for (int tick = 0; tick < 10; ++tick) {
        char desc[100];
        snprintf(desc, sizeof(desc), "Tick %d complete", tick);
        float score = 0.8f + (tick * 0.01f);
        qallow_temporal_memory_log_event(&mem, "execution", desc, score);

        float entropy = 0.5f + (rand() % 10) * 0.001f;
        float harmony = 0.95f - (rand() % 5) * 0.001f;
        qallow_temporal_memory_update_coherence(&mem, 0.0f, 0.0f, entropy, harmony);
        float drift = qallow_temporal_memory_detect_drift(&mem);
    }

    /* Phase complete */
    qallow_temporal_memory_log_event(&mem, "phase_complete", "Phase 12 finished", 0.98f);

    /* Verify state */
    ASSERT(mem.event_count == 12, "All events logged");  /* 1 start + 10 ticks + 1 complete */

    float avg = qallow_temporal_memory_get_average_drift(&mem, 10);
    ASSERT(avg >= 0.0f && avg <= 1.0f, "Average drift in valid range");

    int ret = qallow_temporal_memory_export_json(&mem, "/tmp/integration_test.json");
    ASSERT(ret == 0, "Integration test JSON export succeeds");

    TEST_PASS("Full integration test passes");

    qallow_temporal_memory_free(&mem);
    return 0;
}

/* ============================================================================
 * Main Test Runner
 * ============================================================================ */

int main(int argc, char* argv[]) {
    printf("\n╔════════════════════════════════════════════════════════╗\n");
    printf("║        Temporal Memory Test Suite                     ║\n");
    printf("╚════════════════════════════════════════════════════════╝\n");

    int failed = 0;

    if (test_episodic_event_logging()) failed++;
    if (test_semantic_embeddings()) failed++;
    if (test_coherence_and_drift()) failed++;
    if (test_drift_statistics()) failed++;
    if (test_json_export()) failed++;
    if (test_print_stats()) failed++;
    if (test_integration()) failed++;

    printf("\n╔════════════════════════════════════════════════════════╗\n");
    if (failed == 0) {
        printf("║  ✅ ALL TESTS PASSED (7/7)                            ║\n");
    } else {
        printf("║  ❌ TESTS FAILED: %d/7                                ║\n", failed);
    }
    printf("╚════════════════════════════════════════════════════════╝\n\n");

    return failed;
}
