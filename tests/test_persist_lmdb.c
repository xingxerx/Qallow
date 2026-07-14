#include "qallow/persist_lmdb.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#ifdef _WIN32
#include <direct.h>
#define TEST_MKDIR(path) _mkdir(path)
#else
#include <sys/stat.h>
#define TEST_MKDIR(path) mkdir(path, 0755)
#endif

static qsw_envelope mk(const uint8_t node_id[QSW_NODE_ID_LEN], uint64_t lam,
                        const char *k, const char *v, uint16_t flags) {
    qsw_envelope e = {0};
    memcpy(e.node_id, node_id, QSW_NODE_ID_LEN);
    e.lamport = lam; e.flags = flags;
    e.key = k; e.key_len = (uint32_t)strlen(k);
    e.blob = v; e.blob_len = (uint32_t)strlen(v);
    return e;
}

int main(void) {
    const char *dir = "build/.test_persist_dir";
    TEST_MKDIR("build");
    TEST_MKDIR(dir); /* ignore EEXIST: LMDB just needs the dir to be present */

    ql_persist_store *store = NULL;
    assert(ql_persist_open(dir, &store) == QLP_OK);

    uint8_t idA[QSW_NODE_ID_LEN], idB[QSW_NODE_ID_LEN];
    memset(idA, 0xAA, sizeof idA);
    memset(idB, 0xBB, sizeof idB);

    bool applied;
    qsw_envelope e1 = mk(idA, 5, "k1", "v1-A", 0);
    assert(ql_persist_merge_blob(store, &e1, &applied) == QLP_OK && applied);

    char buf[64]; uint32_t len;
    assert(ql_persist_get(store, "k1", 2, buf, sizeof buf, &len) == QLP_OK);
    assert(len == 4 && !memcmp(buf, "v1-A", 4));

    /* stale write must lose */
    qsw_envelope e2 = mk(idB, 3, "k1", "stale", 0);
    assert(ql_persist_merge_blob(store, &e2, &applied) == QLP_OK && !applied);
    assert(ql_persist_get(store, "k1", 2, buf, sizeof buf, &len) == QLP_OK);
    assert(len == 4 && !memcmp(buf, "v1-A", 4));

    /* tie, higher node id wins: B > A */
    qsw_envelope e3 = mk(idB, 5, "k1", "v1-B", 0);
    assert(ql_persist_merge_blob(store, &e3, &applied) == QLP_OK && applied);
    assert(ql_persist_get(store, "k1", 2, buf, sizeof buf, &len) == QLP_OK);
    assert(len == 4 && !memcmp(buf, "v1-B", 4));

    /* tie, lower node id loses: A < B */
    qsw_envelope e4 = mk(idA, 5, "k1", "v1-A2", 0);
    assert(ql_persist_merge_blob(store, &e4, &applied) == QLP_OK && !applied);

    /* tombstone wins and deletes */
    qsw_envelope e5 = mk(idB, 6, "k1", "", 0x1);
    assert(ql_persist_merge_blob(store, &e5, &applied) == QLP_OK && applied);
    assert(ql_persist_get(store, "k1", 2, buf, sizeof buf, &len) == QLP_E_IO);

    /* fresh key */
    qsw_envelope e6 = mk(idA, 1, "k2", "hello", 0);
    assert(ql_persist_merge_blob(store, &e6, &applied) == QLP_OK && applied);
    assert(ql_persist_get(store, "k2", 2, buf, sizeof buf, &len) == QLP_OK);
    assert(len == 5 && !memcmp(buf, "hello", 5));

    ql_persist_close(store);
    puts("persist_lmdb: all tests passed (LWW merge, tie-break, tombstone, get)");
    return 0;
}
