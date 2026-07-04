/* qallow/sync_wire.h
 * Wire format for Qallow mesh sync (Task 1).
 * Transport-agnostic: feed bytes in, get frames out. No sockets here.
 * Layout: hand-rolled, length-prefixed, little-endian, explicit
 * serialization (no struct casts; immune to padding/endianness).
 *
 * HARD INVARIANT: LIMEN credentials and quantum link traffic must
 * never enter these payloads. This layer carries semantic memory
 * envelopes only.
 *
 * NOTE: envelope field contract defined here pending reconciliation
 * with local persist_lmdb.h (Task 0 not yet pushed to remote).
 */
#ifndef QALLOW_SYNC_WIRE_H
#define QALLOW_SYNC_WIRE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define QSW_MAGIC        0x4E595351u /* "QSYN" LE */
#define QSW_PROTO_VER    1u
#define QSW_NODE_ID_LEN  16u
#define QSW_MAX_KEY_LEN  1024u
#define QSW_MAX_BLOB_LEN (16u * 1024u * 1024u) /* 16 MiB per envelope */

/* Frame types */
typedef enum {
    QSW_F_HELLO     = 1, /* handshake: magic, proto ver, node id, lamport */
    QSW_F_HELLO_ACK = 2, /* same body as HELLO, sent in reply */
    QSW_F_ENVELOPE  = 3, /* one sync envelope */
    QSW_F_BATCH_END = 4, /* sender finished a sync pass (carries lamport) */
    QSW_F_BYE       = 5  /* orderly close */
} qsw_frame_type;

/* Error codes (negative), success >= 0 */
typedef enum {
    QSW_OK          = 0,
    QSW_NEED_MORE   = 1,  /* decoder: partial frame, feed more bytes */
    QSW_E_ARG       = -1,
    QSW_E_SPACE     = -2, /* output buffer too small */
    QSW_E_MAGIC     = -3,
    QSW_E_VERSION   = -4,
    QSW_E_MALFORMED = -5,
    QSW_E_TOO_BIG   = -6
} qsw_status;

/* Handshake body (HELLO / HELLO_ACK / BATCH_END uses lamport only) */
typedef struct {
    uint32_t magic;
    uint16_t proto_ver;
    uint16_t caps;                        /* capability flags, 0 for now */
    uint8_t  node_id[QSW_NODE_ID_LEN];
    uint64_t lamport;                     /* sender clock at send time */
} qsw_hello;

/* Sync envelope. Mirrors the Task 0 versioned envelope:
 * origin node, Lamport clock, schema version, key, value blob. */
typedef struct {
    uint8_t     node_id[QSW_NODE_ID_LEN]; /* origin node */
    uint64_t    lamport;                  /* origin Lamport timestamp */
    uint16_t    schema_ver;               /* envelope schema version */
    uint16_t    flags;                    /* 0x1 = tombstone */
    uint32_t    key_len;
    uint32_t    blob_len;
    const void *key;                      /* borrowed, not owned */
    const void *blob;                     /* borrowed, not owned */
} qsw_envelope;

/* Decoded frame */
typedef struct {
    uint8_t type;                         /* qsw_frame_type */
    union {
        qsw_hello    hello;               /* HELLO / HELLO_ACK */
        qsw_envelope env;                 /* ENVELOPE */
        uint64_t     lamport;             /* BATCH_END */
    } u;
} qsw_frame;

/* --- Encoding ---------------------------------------------------
 * Each writes one complete frame into out. Returns bytes written,
 * or negative qsw_status. Frame layout on the wire:
 *   u8  type | u32 body_len | body
 */
int32_t qsw_encode_hello(uint8_t *out, size_t cap, uint8_t type,
                         const uint8_t node_id[QSW_NODE_ID_LEN],
                         uint64_t lamport);
int32_t qsw_encode_envelope(uint8_t *out, size_t cap,
                            const qsw_envelope *env);
int32_t qsw_encode_batch_end(uint8_t *out, size_t cap, uint64_t lamport);
int32_t qsw_encode_bye(uint8_t *out, size_t cap);

/* --- Decoding ----------------------------------------------------
 * Incremental: works with partial reads from any transport.
 * Feed a byte window; on QSW_OK, *frame is valid (env.key/env.blob
 * point INTO buf; copy before reusing buf) and *consumed is set.
 * On QSW_NEED_MORE, keep the bytes and feed a longer window.
 */
qsw_status qsw_decode(const uint8_t *buf, size_t len,
                      qsw_frame *frame, size_t *consumed);

/* Handshake validation: checks magic + proto version. */
qsw_status qsw_hello_validate(const qsw_hello *h);

#ifdef __cplusplus
}
#endif
#endif /* QALLOW_SYNC_WIRE_H */
