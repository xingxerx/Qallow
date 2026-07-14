/* src/mind/persist_lmdb.c
 * Task 0: LMDB-backed store for sync envelopes. See
 * include/qallow/persist_lmdb.h for the merge contract.
 *
 * On-disk record layout (value blob for a key), explicit and
 * little-endian so a store is portable across hosts of either
 * endianness, matching the wire format's own convention:
 *   u64 lamport | u8[QSW_NODE_ID_LEN] node_id | u16 flags | blob
 */
#include "qallow/persist_lmdb.h"
#include "lmdb.h"

#include <stdlib.h>
#include <string.h>

#define QLP_REC_HDR_LEN (8u + QSW_NODE_ID_LEN + 2u)
#define QLP_MAPSIZE      ((size_t)1 << 30) /* 1 GiB address space, lazily paged */

struct ql_persist_store {
    MDB_env *env;
};

static void put_u64le(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i));
}
static uint64_t get_u64le(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)p[i] << (8 * i);
    return v;
}
static void put_u16le(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8);
}
static uint16_t get_u16le(const uint8_t *p) {
    return (uint16_t)(p[0] | (p[1] << 8));
}

qlp_status ql_persist_open(const char *dir_path, ql_persist_store **out) {
    if (!dir_path || !out) return QLP_E_ARG;

    ql_persist_store *store = (ql_persist_store *)calloc(1, sizeof(*store));
    if (!store) return QLP_E_OPEN;

    if (mdb_env_create(&store->env) != 0) {
        free(store);
        return QLP_E_OPEN;
    }
    if (mdb_env_set_mapsize(store->env, QLP_MAPSIZE) != 0 ||
        mdb_env_open(store->env, dir_path, 0, 0664) != 0) {
        mdb_env_close(store->env);
        free(store);
        return QLP_E_OPEN;
    }

    *out = store;
    return QLP_OK;
}

void ql_persist_close(ql_persist_store *store) {
    if (!store) return;
    mdb_env_close(store->env);
    free(store);
}

qlp_status ql_persist_merge_blob(ql_persist_store *store,
                                 const qsw_envelope *env,
                                 bool *out_applied) {
    if (out_applied) *out_applied = false;
    if (!store || !env || (!env->key && env->key_len) ||
        (!env->blob && env->blob_len))
        return QLP_E_ARG;
    if (env->key_len > QSW_MAX_KEY_LEN || env->blob_len > QSW_MAX_BLOB_LEN)
        return QLP_E_TOO_BIG;

    bool tombstone = (env->flags & 0x1u) != 0;

    MDB_txn *txn = NULL;
    if (mdb_txn_begin(store->env, NULL, 0, &txn) != 0) return QLP_E_TXN;

    MDB_dbi dbi;
    if (mdb_dbi_open(txn, NULL, MDB_CREATE, &dbi) != 0) {
        mdb_txn_abort(txn);
        return QLP_E_OPEN;
    }

    MDB_val k, v;
    k.mv_size = env->key_len;
    k.mv_data = (void *)env->key;

    int get_rc = mdb_get(txn, dbi, &k, &v);
    if (get_rc == 0) {
        const uint8_t *rec = (const uint8_t *)v.mv_data;
        uint64_t cur_lamport = get_u64le(rec);
        const uint8_t *cur_node = rec + 8;

        if (env->lamport < cur_lamport ||
            (env->lamport == cur_lamport &&
             memcmp(env->node_id, cur_node, QSW_NODE_ID_LEN) <= 0)) {
            mdb_txn_abort(txn);
            return QLP_OK; /* stale or tied-and-losing: silent no-op */
        }
    } else if (get_rc != MDB_NOTFOUND) {
        mdb_txn_abort(txn);
        return QLP_E_IO;
    }

    int put_rc;
    if (tombstone) {
        put_rc = mdb_del(txn, dbi, &k, NULL);
        if (put_rc != 0 && put_rc != MDB_NOTFOUND) {
            mdb_txn_abort(txn);
            return QLP_E_IO;
        }
    } else {
        size_t rec_len = QLP_REC_HDR_LEN + env->blob_len;
        uint8_t *rec = (uint8_t *)malloc(rec_len);
        if (!rec) {
            mdb_txn_abort(txn);
            return QLP_E_IO;
        }
        put_u64le(rec, env->lamport);
        memcpy(rec + 8, env->node_id, QSW_NODE_ID_LEN);
        put_u16le(rec + 8 + QSW_NODE_ID_LEN, env->flags);
        if (env->blob_len) memcpy(rec + QLP_REC_HDR_LEN, env->blob, env->blob_len);

        v.mv_size = rec_len;
        v.mv_data = rec;
        put_rc = mdb_put(txn, dbi, &k, &v, 0);
        free(rec);
        if (put_rc != 0) {
            mdb_txn_abort(txn);
            return QLP_E_IO;
        }
    }

    if (mdb_txn_commit(txn) != 0) return QLP_E_TXN;
    if (out_applied) *out_applied = true;
    return QLP_OK;
}

qlp_status ql_persist_get(ql_persist_store *store,
                          const void *key, uint32_t key_len,
                          void *out_val, uint32_t cap, uint32_t *out_len) {
    if (!store || !key || !out_len) return QLP_E_ARG;

    MDB_txn *txn = NULL;
    if (mdb_txn_begin(store->env, NULL, MDB_RDONLY, &txn) != 0) return QLP_E_TXN;

    MDB_dbi dbi;
    if (mdb_dbi_open(txn, NULL, 0, &dbi) != 0) {
        mdb_txn_abort(txn);
        return QLP_E_IO;
    }

    MDB_val k, v;
    k.mv_size = key_len;
    k.mv_data = (void *)key;

    int rc = mdb_get(txn, dbi, &k, &v);
    if (rc != 0) {
        mdb_txn_abort(txn);
        return QLP_E_IO;
    }
    if (v.mv_size < QLP_REC_HDR_LEN) {
        mdb_txn_abort(txn);
        return QLP_E_IO;
    }

    const uint8_t *rec = (const uint8_t *)v.mv_data;
    uint16_t flags = get_u16le(rec + 8 + QSW_NODE_ID_LEN);
    if (flags & 0x1u) { /* tombstoned */
        mdb_txn_abort(txn);
        return QLP_E_IO;
    }

    uint32_t blob_len = (uint32_t)(v.mv_size - QLP_REC_HDR_LEN);
    *out_len = blob_len;
    if (out_val && cap) {
        uint32_t n = blob_len < cap ? blob_len : cap;
        memcpy(out_val, rec + QLP_REC_HDR_LEN, n);
    }

    mdb_txn_abort(txn); /* read-only: abort == cheap close */
    return QLP_OK;
}
