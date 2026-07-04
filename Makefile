# Qallow C targets. Rust workspace builds via `cargo build`; this file
# covers the standalone C modules and their tests.
CC      ?= cc
CFLAGS  ?= -std=c11 -Wall -Wextra -Werror -O2
CPPFLAGS += -Iinclude
BUILDDIR := build

TESTS := $(BUILDDIR)/test_sync_wire

.PHONY: test clean

test: $(TESTS)
	@set -e; for t in $(TESTS); do ./$$t; done

$(BUILDDIR)/test_sync_wire: src/mind/sync_wire.c tests/test_sync_wire.c include/qallow/sync_wire.h | $(BUILDDIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) src/mind/sync_wire.c tests/test_sync_wire.c -o $@

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

clean:
	rm -rf $(BUILDDIR)
