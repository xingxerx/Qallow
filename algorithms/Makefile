CC := gcc
CFLAGS := -std=c11 -O2 -Wall -Wextra -I../core/include
LDFLAGS := -lm

# Core ethics library
CORE_SOURCES := ethics_core.c ethics_learn.c ethics_bayes.c ethics_feed.c
CORE_OBJECTS := $(CORE_SOURCES:.c=.o)

# Test executables
TEST_SOURCES := ethics_test.c ethics_test_feed.c
TESTS := ethics_test ethics_test_feed

all: $(TESTS)

# Original test (static inputs)
ethics_test: ethics_test.o $(CORE_OBJECTS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# Closed-loop test (hardware signals)
ethics_test_feed: ethics_test_feed.o $(CORE_OBJECTS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(CORE_OBJECTS) ethics_test.o ethics_test_feed.o $(TESTS)

.PHONY: all clean

