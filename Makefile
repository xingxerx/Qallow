PROJECT := qallow

ACCELERATOR ?= CUDA

CC ?= gcc
CXX ?= g++
NVCC ?= nvcc
BUILD_DIR ?= build/$(ACCELERATOR)

ifeq ($(ACCELERATOR),CPU)
CUDA_ENABLED := 0
BIN ?= $(BUILD_DIR)/qallow_unified_cpu
else ifeq ($(ACCELERATOR),CUDA)
CUDA_ENABLED := 1
BIN ?= $(BUILD_DIR)/qallow_unified_cuda
else
$(error Invalid ACCELERATOR '$(ACCELERATOR)'. Expected CPU or CUDA.)
endif

INCS :=
INCS += -I. -Icore/include -Iruntime -Iinclude -Iethics -I/usr/local/cuda/include -I/opt/cuda/targets/x86_64-linux/include

GENCODE ?= -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90

C_DEFINES := -D_POSIX_C_SOURCE=200809L -D_DEFAULT_SOURCE -DCUDA_ENABLED=$(CUDA_ENABLED)
CFLAGS ?= -O2 -Wall -Wextra -std=c11 $(C_DEFINES) $(INCS)
CXXFLAGS ?= -O2 -Wall -Wextra -std=c++17 $(C_DEFINES) $(INCS)
CUFLAGS ?= -O2 $(INCS) $(GENCODE) -Xcompiler "-Wall -Wextra" -DCUDA_ENABLED=$(CUDA_ENABLED)

LDFLAGS ?=
LINK_LIBS := -lm

LIBTORCH_HOME ?=
LIBTORCH_LIB ?=

ifdef USE_LIBTORCH
INCS += -I$(LIBTORCH_HOME)/include -I$(LIBTORCH_HOME)/include/torch/csrc/api/include
CXXFLAGS += -DUSE_LIBTORCH
ifeq ($(LIBTORCH_LIB),)
LIBTORCH_LIB := $(LIBTORCH_HOME)/lib
endif
LINK_LIBS += -L$(LIBTORCH_LIB) -Wl,-rpath,$(LIBTORCH_LIB)
LINK_LIBS += -ltorch_cpu -ltorch -lc10
ifdef USE_LIBTORCH_CUDA
LINK_LIBS += -ltorch_cuda
endif
endif

ifeq ($(CUDA_ENABLED),1)
LINK_LIBS += -lcudart -lcurand
LINKER ?= $(NVCC)
else
LINKER ?= $(CXX)
endif

SRC_C := $(filter-out interface/qallow_ui.c,$(wildcard interface/*.c)) \
         $(wildcard backend/cpu/*.c) \
         $(wildcard io/adapters/*.c) \
         $(wildcard src/cli/*.c) \
         $(wildcard src/distributed/*.c) \
         $(wildcard src/ethics/*.c) \
		 $(wildcard src/quantum/*.c) \
         algorithms/ethics_core.c \
         algorithms/ethics_learn.c \
         algorithms/ethics_bayes.c \
         runtime/meta_introspect.c \
         $(wildcard src/mind/*.c) \
         src/qallow_phase13.c
SRC_CPP := $(wildcard src/runtime/*.cpp) \
           runtime/dl_integration.cpp
SRC_CU := $(wildcard backend/cuda/*.cu)

ifeq ($(CUDA_ENABLED),0)
SRC_CU :=
endif

OBJ_C := $(SRC_C:%.c=$(BUILD_DIR)/%.o)
OBJ_CPP := $(SRC_CPP:%.cpp=$(BUILD_DIR)/%.o)
OBJ_CU := $(SRC_CU:%.cu=$(BUILD_DIR)/%.o)

.PHONY: all clean bench profile

all: $(BIN)

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(if $(filter src/qallow_phase13.c,$<),-DQALLOW_PHASE13_EMBEDDED,) -c $< -o $@

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(CUFLAGS) -c $< -o $@

$(BIN): $(OBJ_C) $(OBJ_CPP) $(OBJ_CU)
	@mkdir -p $(dir $(BIN))
	$(LINKER) $(OBJ_C) $(OBJ_CPP) $(OBJ_CU) $(LDFLAGS) -o $(BIN) $(LINK_LIBS)

RUN_ARGS := run
ifeq ($(CUDA_ENABLED),1)
RUN_ARGS += --accelerator
endif

bench: $(BIN)
	$(BIN) $(RUN_ARGS) --bench

NSIGHT ?= nv-nsight-cu-cli
ifeq ($(CUDA_ENABLED),1)
profile: $(BIN)
	@if command -v $(NSIGHT) >/dev/null 2>&1; then \
	  NSIGHT_CMD=$(NSIGHT); \
	elif [ -x /opt/cuda/NsightCompute/nv-nsight-cu-cli ]; then \
	  NSIGHT_CMD=/opt/cuda/NsightCompute/nv-nsight-cu-cli; \
	else \
	  echo "nv-nsight-cu-cli not found"; exit 1; \
	fi; \
	$$NSIGHT_CMD --set full --target-processes all -- $(BIN) $(RUN_ARGS) --bench
else
profile:
	@echo "Profile target requires ACCELERATOR=CUDA"
endif

clean:
	rm -rf $(BUILD_DIR)
