# Qallow C targets. Rust workspace builds via `cargo build`; this file
# covers the standalone C modules and their tests.
CC      ?= cc
CFLAGS  ?= -std=c11 -Wall -Wextra -Werror -O2
# MDB_MAXKEYSIZE matches QSW_MAX_KEY_LEN (sync_wire.h); LMDB's default
# (511) is smaller and would reject valid envelopes.
CPPFLAGS += -Iinclude -Ithird_party/lmdb -DMDB_MAXKEYSIZE=1024
BUILDDIR := build

TESTS := $(BUILDDIR)/test_sync_wire $(BUILDDIR)/test_persist_lmdb

.PHONY: test clean

test: $(TESTS)
	@set -e; for t in $(TESTS); do ./$$t; done

$(BUILDDIR)/test_sync_wire: src/mind/sync_wire.c tests/test_sync_wire.c include/qallow/sync_wire.h | $(BUILDDIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) src/mind/sync_wire.c tests/test_sync_wire.c -o $@

$(BUILDDIR)/test_persist_lmdb: src/mind/persist_lmdb.c tests/test_persist_lmdb.c \
		include/qallow/persist_lmdb.h include/qallow/sync_wire.h \
		third_party/lmdb/mdb.c third_party/lmdb/midl.c | $(BUILDDIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) src/mind/persist_lmdb.c tests/test_persist_lmdb.c \
		third_party/lmdb/mdb.c third_party/lmdb/midl.c -o $@

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

clean:
	rm -rf $(BUILDDIR)
