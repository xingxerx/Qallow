CC := gcc
CFLAGS := -std=c11 -O2 -Wall -Wextra -I../core/include
LDFLAGS := -lm

ETHICS_SRC := ../algorithms/ethics_core.c ../algorithms/ethics_learn.c \
              ../algorithms/ethics_bayes.c ../algorithms/ethics_feed.c

all: qallow_ethics_integration

qallow_ethics_integration: qallow_ethics_integration.c $(ETHICS_SRC)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f qallow_ethics_integration

.PHONY: all clean
