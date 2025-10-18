# Native C + CUDA (no VM)
PROJECT = qallow
CC = gcc
NVCC = nvcc
CFLAGS = -O2 -Wall -Iinclude
LDFLAGS = -lm
CUFLAGS = -O2 -arch=sm_89 -Iinclude
CULIBS = -lcurand

SRC_C = src/qallow_kernel.c src/overlays.c
SRC_CU = emulation/photonic.cu emulation/quantum.cu

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

