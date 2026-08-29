Sovereignty map for the persist/sync core loop

- sync wire
  - file: src/mind/sync_wire.c
  - dependency: none
- persist store
  - file: src/mind/persist_lmdb.c
  - dependency: vendored LMDB only
- storage engine
  - files: third_party/lmdb/mdb.c, third_party/lmdb/midl.c, third_party/lmdb/lmdb.h
  - dependency: LMDB (vendored)

Notes
- No Gemma, Ollama, FastAPI, or external reasoning surfaces are invoked by the persist/sync core loop.
- All invariants at the persist/sync gate are enforced in C before LMDB writes.
