PROJECT := qallow
BIN ?= build/qallow_unified_cuda

CC ?= gcc
CXX ?= g++
NVCC ?= nvcc

INCS :=
INCS += -I. -Icore/include -Iinclude -Iethics -Iruntime -I/usr/local/cuda/include -I/opt/cuda/targets/x86_64-linux/include

GENCODE ?= -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90

CFLAGS ?= -O2 -Wall -std=c11 -DCUDA_ENABLED=1 $(INCS)
CXXFLAGS ?= -O2 -Wall -std=c++17 -DCUDA_ENABLED=1 $(INCS)
CUFLAGS ?= -O2 $(INCS) $(GENCODE) -Xcompiler "-Wall" -DCUDA_ENABLED=1

LDFLAGS ?= -lm
CULIBS ?= -lcudart -lcurand -lm

BUILD_DIR ?= build

SRC_C := core/main.c \
		 runtime/meta_introspect.c \
		 algorithms/ethics_core.c
SRC_CU := backend/cuda/p12_elasticity.cu \
          backend/cuda/p13_harmonic.cu \
          backend/cuda/phase16_meta_introspect.cu

OBJ_C := $(SRC_C:%.c=$(BUILD_DIR)/%.o)
OBJ_CU := $(SRC_CU:%.cu=$(BUILD_DIR)/%.o)

.PHONY: all clean bench profile

all: $(BIN)

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(CUFLAGS) -c $< -o $@

$(BIN): $(OBJ_C) $(OBJ_CU)
	@mkdir -p $(dir $(BIN))
	$(NVCC) $(OBJ_C) $(OBJ_CU) -o $(BIN) $(CULIBS)

bench: $(BIN)
	$(BIN) run --accelerator --bench

NSIGHT ?= nv-nsight-cu-cli
profile: $(BIN)
	@if command -v $(NSIGHT) >/dev/null 2>&1; then \
	  NSIGHT_CMD=$(NSIGHT); \
	elif [ -x /opt/cuda/NsightCompute/nv-nsight-cu-cli ]; then \
	  NSIGHT_CMD=/opt/cuda/NsightCompute/nv-nsight-cu-cli; \
	else \
	  echo "nv-nsight-cu-cli not found"; exit 1; \
	fi; \
	$$NSIGHT_CMD --set full --target-processes all -- $(BIN) run --accelerator --bench

clean:
	rm -rf $(BUILD_DIR)
