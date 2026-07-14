/* qallow/persist_lmdb.h
 * Task 0: LMDB-backed persistence for sync envelopes (see sync_wire.h).
 * Reuses qsw_envelope as the merge input so the wire contract and the
 * on-disk merge contract stay reconciled by construction.
 */
#ifndef QALLOW_PERSIST_LMDB_H
#define QALLOW_PERSIST_LMDB_H

#include "qallow/sync_wire.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ql_persist_store ql_persist_store; /* opaque */

typedef enum {
    QLP_OK        = 0,
    QLP_E_ARG     = -1,
    QLP_E_OPEN    = -2, /* LMDB env/db open failed */
    QLP_E_TXN     = -3, /* LMDB transaction begin/commit failed */
    QLP_E_IO      = -4, /* LMDB read/write op failed, or key not found */
    QLP_E_TOO_BIG = -5  /* key/blob exceeds QSW_MAX_KEY_LEN/QSW_MAX_BLOB_LEN */
} qlp_status;

/* Opens (creating if absent) an LMDB-backed store rooted at dir_path.
 * dir_path must already exist as a directory. *out is set on success
 * and must be released with ql_persist_close(). */
qlp_status ql_persist_open(const char *dir_path, ql_persist_store **out);
void       ql_persist_close(ql_persist_store *store);

/* Merges one sync envelope into the store, last-writer-wins:
 *   - higher env->lamport wins outright.
 *   - on a tie, the envelope whose node_id compares greater
 *     (memcmp over QSW_NODE_ID_LEN) wins.
 *   - a losing envelope is a silent no-op, not an error.
 * A winning envelope with (flags & 0x1) set (tombstone) deletes the
 * key instead of writing env->blob.
 *
 * On QLP_OK, *out_applied reports whether the store was modified
 * (true) or the envelope was discarded as stale/tied-and-losing
 * (false). out_applied may be NULL if the caller doesn't need it.
 */
qlp_status ql_persist_merge_blob(ql_persist_store *store,
                                 const qsw_envelope *env,
                                 bool *out_applied);

/* Fetches the current value for key into a caller-provided buffer.
 * On QLP_OK, *out_len is the stored value's length; at most cap bytes
 * are copied into out_val (out_len may exceed cap on a short buffer).
 * Returns QLP_E_IO if key is absent or tombstoned. */
qlp_status ql_persist_get(ql_persist_store *store,
                          const void *key, uint32_t key_len,
                          void *out_val, uint32_t cap, uint32_t *out_len);

#ifdef __cplusplus
}
#endif
#endif /* QALLOW_PERSIST_LMDB_H */
