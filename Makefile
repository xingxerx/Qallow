PROJECT = qallow
CC = gcc
NVCC = nvcc
SM ?= sm_89

CFLAGS = -O2 -Wall -std=c11 -Icore/include
LDFLAGS = -lm
CUFLAGS = -O2 -std=c++14 -Icore/include -arch=$(SM)
CULIBS = -lcurand

SRC_C = $(wildcard backend/cpu/*.c)
SRC_CU = $(wildcard backend/cuda/*.cu)

OBJ_C = $(SRC_C:.c=.o)
OBJ_CU = $(SRC_CU:.cu=.o)

all: $(PROJECT)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(CUFLAGS) -c $< -o $@

$(PROJECT): $(OBJ_C) $(OBJ_CU)
	$(NVCC) -o $@ $^ $(CULIBS) $(LDFLAGS)

clean:
	rm -f $(OBJ_C) $(OBJ_CU) $(PROJECT)
