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
	$(NVCC) -o build/qallow_unified_cuda \
	  backend/cpu/adaptive.o backend/cpu/chronometric.o backend/cpu/error_handler.o \
	  backend/cpu/ethics.o backend/cpu/goal_synthesizer.o backend/cpu/govern.o \
	  backend/cpu/ingest.o backend/cpu/lattice_integration.o backend/cpu/multi_pocket.o \
	  backend/cpu/overlay.o backend/cpu/phase12_elasticity.o backend/cpu/phase13_harmonic.o \
	  backend/cpu/phase14_coherence.o backend/cpu/phase14_entanglement.o \
	  backend/cpu/phase15_convergence.o backend/cpu/phase15_singularity.o \
	  backend/cpu/phase7_core.o backend/cpu/pocket.o backend/cpu/pocket_dimension.o \
	  backend/cpu/ppai.o backend/cpu/qallow_kernel.o backend/cpu/qcp.o \
	  backend/cpu/self_reflection.o backend/cpu/semantic_memory.o backend/cpu/telemetry.o \
	  backend/cpu/transfer_engine.o backend/cpu/verify.o \
	  backend/cuda/photonic.o backend/cuda/pocket.o backend/cuda/ppai_kernels.o \
	  backend/cuda/qcp_kernels.o backend/cuda/quantum.o \
	  core/main.c \
	  -lcurand -lm

clean:
	rm -f $(OBJ_C) $(OBJ_CU) $(PROJECT)
