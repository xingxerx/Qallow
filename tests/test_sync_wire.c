/* tests/test_sync_wire.c
 * Two in-process nodes, separate in-memory stores (stand-in for two
 * LMDB dirs until Task 0 lands on remote). Node A streams envelopes
 * to node B over a byte pipe; B merges last-writer-wins by Lamport,
 * origin node id as tiebreak — matching ql_persist_merge_blob semantics.
 * Also exercises partial-read decoding (1-byte drip feed).
 */
#include "qallow/sync_wire.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

#define STORE_MAX 16
typedef struct {
    char     key[64];
    char     val[128];
    uint64_t lamport;
    uint8_t  origin[QSW_NODE_ID_LEN];
    int      used;
} rec;
typedef struct { rec r[STORE_MAX]; uint64_t clock; } store;

static int node_cmp(const uint8_t *a, const uint8_t *b) {
    return memcmp(a, b, QSW_NODE_ID_LEN);
}
/* LWW merge: higher lamport wins; tie -> higher node id wins */
static void store_merge(store *s, const qsw_envelope *e) {
    if (e->lamport > s->clock) s->clock = e->lamport; /* Lamport recv rule */
    for (int i = 0; i < STORE_MAX; i++) {
        rec *r = &s->r[i];
        if (r->used && strlen(r->key) == e->key_len &&
            !memcmp(r->key, e->key, e->key_len)) {
            if (e->lamport < r->lamport) return;
            if (e->lamport == r->lamport &&
                node_cmp(e->node_id, r->origin) <= 0) return;
            memcpy(r->val, e->blob, e->blob_len); r->val[e->blob_len] = 0;
            r->lamport = e->lamport;
            memcpy(r->origin, e->node_id, QSW_NODE_ID_LEN);
            return;
        }
    }
    for (int i = 0; i < STORE_MAX; i++) {
        rec *r = &s->r[i];
        if (!r->used) {
            memcpy(r->key, e->key, e->key_len); r->key[e->key_len] = 0;
            memcpy(r->val, e->blob, e->blob_len); r->val[e->blob_len] = 0;
            r->lamport = e->lamport;
            memcpy(r->origin, e->node_id, QSW_NODE_ID_LEN);
            r->used = 1;
            return;
        }
    }
    assert(0 && "store full");
}
static const char *store_get(store *s, const char *k) {
    for (int i = 0; i < STORE_MAX; i++)
        if (s->r[i].used && !strcmp(s->r[i].key, k)) return s->r[i].val;
    return NULL;
}

static int32_t enc_env(uint8_t *out, size_t cap,
                       const uint8_t id[QSW_NODE_ID_LEN], uint64_t lam,
                       const char *k, const char *v) {
    qsw_envelope e = {0};
    memcpy(e.node_id, id, QSW_NODE_ID_LEN);
    e.lamport = lam; e.schema_ver = 1;
    e.key = k;  e.key_len  = (uint32_t)strlen(k);
    e.blob = v; e.blob_len = (uint32_t)strlen(v);
    return qsw_encode_envelope(out, cap, &e);
}

int main(void) {
    uint8_t idA[QSW_NODE_ID_LEN], idB[QSW_NODE_ID_LEN];
    memset(idA, 0xAA, sizeof idA);
    memset(idB, 0xBB, sizeof idB);
    store A = {0}, B = {0};
    A.clock = 7; B.clock = 3;

    /* --- pipe: A -> B ------------------------------------------ */
    uint8_t pipe[4096]; size_t n = 0; int32_t w;

    w = qsw_encode_hello(pipe + n, sizeof pipe - n, QSW_F_HELLO, idA, A.clock);
    assert(w > 0); n += (size_t)w;
    w = enc_env(pipe + n, sizeof pipe - n, idA, 5, "dream/41", "lucid:false");
    assert(w > 0); n += (size_t)w;
    w = enc_env(pipe + n, sizeof pipe - n, idA, 7, "phase12/target", "mesh");
    assert(w > 0); n += (size_t)w;
    /* conflict record: B already holds key "shared" at lamport 6 */
    {
        qsw_envelope pre = {0};
        memcpy(pre.node_id, idB, QSW_NODE_ID_LEN);
        pre.lamport = 6; pre.key = "shared"; pre.key_len = 6;
        pre.blob = "from-B"; pre.blob_len = 6;
        store_merge(&B, &pre);
    }
    w = enc_env(pipe + n, sizeof pipe - n, idA, 4, "shared", "stale-from-A");
    assert(w > 0); n += (size_t)w; /* must lose: lamport 4 < 6 */
    w = qsw_encode_batch_end(pipe + n, sizeof pipe - n, A.clock);
    assert(w > 0); n += (size_t)w;
    w = qsw_encode_bye(pipe + n, sizeof pipe - n);
    assert(w > 0); n += (size_t)w;
    /* also include an ACK from B to verify frame acceptance */
    w = qsw_encode_hello(pipe + n, sizeof pipe - n, QSW_F_HELLO_ACK, idB, B.clock);
    assert(w > 0); n += (size_t)w;

    /* --- B decodes with 1-byte drip feed (partial reads) ------- */
    uint8_t rx[4096]; size_t rxn = 0, off = 0;
    int frames = 0, got_hello = 0, got_bye = 0, got_ack = 0;
    for (size_t i = 0; i < n; i++) {
        rx[rxn++] = pipe[i]; /* transport delivers one byte at a time */
        for (;;) {
            qsw_frame f; size_t used;
            qsw_status st = qsw_decode(rx + off, rxn - off, &f, &used);
            if (st == QSW_NEED_MORE) break;
            assert(st == QSW_OK);
            frames++;
            switch (f.type) {
            case QSW_F_HELLO:
                assert(qsw_hello_validate(&f.u.hello) == QSW_OK);
                assert(f.u.hello.lamport == 7);
                got_hello = 1;
                break;
            case QSW_F_HELLO_ACK:
                assert(qsw_hello_validate(&f.u.hello) == QSW_OK);
                assert(f.u.hello.lamport == 6);
                got_ack = 1;
                break;
            case QSW_F_ENVELOPE:
                store_merge(&B, &f.u.env);
                break;
            case QSW_F_BATCH_END:
                assert(f.u.lamport == 7);
                break;
            case QSW_F_BYE:
                got_bye = 1;
                break;
            }
            off += used;
        }
    }
    assert(got_hello && got_bye && got_ack && frames == 7);

    /* --- verify merge results ----------------------------------- */
    assert(!strcmp(store_get(&B, "dream/41"), "lucid:false"));
    assert(!strcmp(store_get(&B, "phase12/target"), "mesh"));
    assert(!strcmp(store_get(&B, "shared"), "from-B")); /* LWW held */
    assert(B.clock >= 7); /* Lamport advanced on receive */

    /* --- malformed input rejected -------------------------------- */
    {
        uint8_t bad[13] = { 99, 8, 0, 0, 0 }; /* unknown type, hdr=5 */
        qsw_frame f; size_t used;
        assert(qsw_decode(bad, sizeof bad, &f, &used) == QSW_E_MALFORMED);
        uint8_t hello[64]; int32_t hl;
        hl = qsw_encode_hello(hello, sizeof hello, QSW_F_HELLO, idA, 1);
        hello[5] ^= 0xFF; /* corrupt magic (body starts after 5-byte hdr) */
        assert(qsw_decode(hello, (size_t)hl, &f, &used) == QSW_OK);
        assert(qsw_hello_validate(&f.u.hello) == QSW_E_MAGIC);
        /* corrupt an ENVELOPE body field so body size no longer matches */
        uint8_t envbuf[256]; int32_t el;
        el = enc_env(envbuf, sizeof envbuf, idA, 8, "k", "v");
        /* flip one bit in key_len (offset: 5 hdr + 16 id + 8 lam + 2 sch + 2 flg) */
        size_t key_len_off = 5 + QSW_NODE_ID_LEN + 8 + 2 + 2;
        envbuf[key_len_off] ^= 0x1;
        assert(qsw_decode(envbuf, (size_t)el, &f, &used) == QSW_E_MALFORMED);
    }

    puts("sync_wire: all tests passed (handshake, drip-feed decode, "
         "LWW merge, malformed rejection)");
    return 0;
}
