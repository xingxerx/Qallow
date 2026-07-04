/* src/mind/sync_wire.c — Qallow mesh sync wire format (Task 1) */
#include "qallow/sync_wire.h"
#include <string.h>

/* ---- LE primitives ---- */
static void put_u16(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8);
}
static void put_u32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)v;         p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}
static void put_u64(uint8_t *p, uint64_t v) {
    put_u32(p, (uint32_t)v); put_u32(p + 4, (uint32_t)(v >> 32));
}
static uint16_t get_u16(const uint8_t *p) {
    return (uint16_t)(p[0] | (uint16_t)p[1] << 8);
}
static uint32_t get_u32(const uint8_t *p) {
    return (uint32_t)p[0]        | (uint32_t)p[1] << 8 |
           (uint32_t)p[2] << 16  | (uint32_t)p[3] << 24;
}
static uint64_t get_u64(const uint8_t *p) {
    return (uint64_t)get_u32(p) | (uint64_t)get_u32(p + 4) << 32;
}

/* Frame header: u8 type + u32 body_len */
#define QSW_HDR 5u
#define HELLO_BODY (4u + 2u + 2u + QSW_NODE_ID_LEN + 8u)
/* Envelope body fixed part: node_id + lamport + schema + flags + lens */
#define ENV_FIXED (QSW_NODE_ID_LEN + 8u + 2u + 2u + 4u + 4u)

static int32_t write_hdr(uint8_t *out, size_t cap, uint8_t type,
                         uint32_t body_len) {
    if (cap < QSW_HDR + body_len) return QSW_E_SPACE;
    out[0] = type;
    put_u32(out + 1, body_len);
    return (int32_t)(QSW_HDR + body_len);
}

int32_t qsw_encode_hello(uint8_t *out, size_t cap, uint8_t type,
                         const uint8_t node_id[QSW_NODE_ID_LEN],
                         uint64_t lamport) {
    if (!out || !node_id) return QSW_E_ARG;
    if (type != QSW_F_HELLO && type != QSW_F_HELLO_ACK) return QSW_E_ARG;
    int32_t total = write_hdr(out, cap, type, HELLO_BODY);
    if (total < 0) return total;
    uint8_t *b = out + QSW_HDR;
    put_u32(b, QSW_MAGIC);
    put_u16(b + 4, QSW_PROTO_VER);
    put_u16(b + 6, 0); /* caps */
    memcpy(b + 8, node_id, QSW_NODE_ID_LEN);
    put_u64(b + 8 + QSW_NODE_ID_LEN, lamport);
    return total;
}

int32_t qsw_encode_envelope(uint8_t *out, size_t cap,
                            const qsw_envelope *env) {
    if (!out || !env || !env->key || (env->blob_len && !env->blob))
        return QSW_E_ARG;
    if (env->key_len == 0 || env->key_len > QSW_MAX_KEY_LEN) return QSW_E_ARG;
    if (env->blob_len > QSW_MAX_BLOB_LEN) return QSW_E_TOO_BIG;
    uint32_t body = ENV_FIXED + env->key_len + env->blob_len;
    int32_t total = write_hdr(out, cap, QSW_F_ENVELOPE, body);
    if (total < 0) return total;
    uint8_t *b = out + QSW_HDR;
    memcpy(b, env->node_id, QSW_NODE_ID_LEN); b += QSW_NODE_ID_LEN;
    put_u64(b, env->lamport);      b += 8;
    put_u16(b, env->schema_ver);   b += 2;
    put_u16(b, env->flags);        b += 2;
    put_u32(b, env->key_len);      b += 4;
    put_u32(b, env->blob_len);     b += 4;
    memcpy(b, env->key, env->key_len); b += env->key_len;
    if (env->blob_len) memcpy(b, env->blob, env->blob_len);
    return total;
}

int32_t qsw_encode_batch_end(uint8_t *out, size_t cap, uint64_t lamport) {
    if (!out) return QSW_E_ARG;
    int32_t total = write_hdr(out, cap, QSW_F_BATCH_END, 8);
    if (total < 0) return total;
    put_u64(out + QSW_HDR, lamport);
    return total;
}

int32_t qsw_encode_bye(uint8_t *out, size_t cap) {
    if (!out) return QSW_E_ARG;
    return write_hdr(out, cap, QSW_F_BYE, 0);
}

qsw_status qsw_hello_validate(const qsw_hello *h) {
    if (!h) return QSW_E_ARG;
    if (h->magic != QSW_MAGIC) return QSW_E_MAGIC;
    if (h->proto_ver != QSW_PROTO_VER) return QSW_E_VERSION;
    return QSW_OK;
}

qsw_status qsw_decode(const uint8_t *buf, size_t len,
                      qsw_frame *frame, size_t *consumed) {
    if (!buf || !frame || !consumed) return QSW_E_ARG;
    if (len < QSW_HDR) return QSW_NEED_MORE;

    uint8_t  type = buf[0];
    uint32_t body = get_u32(buf + 1);
    if (body > QSW_HDR + ENV_FIXED + QSW_MAX_KEY_LEN + QSW_MAX_BLOB_LEN)
        return QSW_E_TOO_BIG;
    if (len < QSW_HDR + (size_t)body) return QSW_NEED_MORE;

    const uint8_t *b = buf + QSW_HDR;
    memset(frame, 0, sizeof *frame);
    frame->type = type;

    switch (type) {
    case QSW_F_HELLO:
    case QSW_F_HELLO_ACK: {
        if (body != HELLO_BODY) return QSW_E_MALFORMED;
        qsw_hello *h = &frame->u.hello;
        h->magic     = get_u32(b);
        h->proto_ver = get_u16(b + 4);
        h->caps      = get_u16(b + 6);
        memcpy(h->node_id, b + 8, QSW_NODE_ID_LEN);
        h->lamport   = get_u64(b + 8 + QSW_NODE_ID_LEN);
        break;
    }
    case QSW_F_ENVELOPE: {
        if (body < ENV_FIXED) return QSW_E_MALFORMED;
        qsw_envelope *e = &frame->u.env;
        memcpy(e->node_id, b, QSW_NODE_ID_LEN); b += QSW_NODE_ID_LEN;
        e->lamport    = get_u64(b); b += 8;
        e->schema_ver = get_u16(b); b += 2;
        e->flags      = get_u16(b); b += 2;
        e->key_len    = get_u32(b); b += 4;
        e->blob_len   = get_u32(b); b += 4;
        if (e->key_len == 0 || e->key_len > QSW_MAX_KEY_LEN ||
            e->blob_len > QSW_MAX_BLOB_LEN ||
            (uint64_t)ENV_FIXED + e->key_len + e->blob_len != body)
            return QSW_E_MALFORMED;
        e->key  = b;
        e->blob = e->blob_len ? b + e->key_len : NULL;
        break;
    }
    case QSW_F_BATCH_END:
        if (body != 8) return QSW_E_MALFORMED;
        frame->u.lamport = get_u64(b);
        break;
    case QSW_F_BYE:
        if (body != 0) return QSW_E_MALFORMED;
        break;
    default:
        return QSW_E_MALFORMED;
    }
    *consumed = QSW_HDR + body;
    return QSW_OK;
}
