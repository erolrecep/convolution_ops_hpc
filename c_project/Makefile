export PATH := /usr/local/cuda/bin:$(PATH)
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
NVCC := $(shell which nvcc 2>/dev/null)

HAS_MONOREPO_C := $(shell if [ -d "../../../_c_" ]; then echo 1; else echo 0; fi)

ifeq ($(HAS_MONOREPO_C),1)
    MONOREPO_INC = -I../../../_c_/include
    MONOREPO_LIB = -L../../../_c_/lib -lc_common
    BUILD_MONOREPO = @$(MAKE) -C ../../../_c_
else
    MONOREPO_INC =
    MONOREPO_LIB =
    BUILD_MONOREPO =
endif

ifeq ($(UNAME_S),Darwin)
    CC ?= clang
    CFLAGS = -O3 -Iinclude $(MONOREPO_INC)
    LIBS = $(MONOREPO_LIB)
    OMP_FLAGS = -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -Xpreprocessor -fopenmp -lomp
else
    CC ?= gcc
    CFLAGS = -O3 -Iinclude $(MONOREPO_INC)
    LIBS = $(MONOREPO_LIB) -lm
    OMP_FLAGS = -fopenmp
endif

BIN_DIR = bin
SRC_DIR = src

TARGETS = $(BIN_DIR)/pure_c $(BIN_DIR)/pthreads_c $(BIN_DIR)/openmp_c $(BIN_DIR)/metal_c $(BIN_DIR)/cuda_c

all: $(BIN_DIR) info $(TARGETS)

info:
	@echo "--- Hardware & Toolchain Capability Check ---"
	@echo "OS: $(UNAME_S)"
	@if [ -n "$(NVCC)" ]; then \
		echo "CUDA Hardware Compiler: Found ($(NVCC)) -> Building full CUDA GPU kernel"; \
	else \
		echo "CUDA Hardware Compiler: Not Found -> Building CUDA capability check stub (Outputs N/A)"; \
	fi
	@if [ "$(UNAME_S)" = "Darwin" ]; then \
		echo "Apple Metal GPU Framework: Found -> Building native Metal MSL GPU compute shader"; \
	else \
		echo "Apple Metal GPU Framework: Not Supported on $(UNAME_S) -> Building Metal check stub (Outputs N/A)"; \
	fi
	@echo "---------------------------------------------"

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/pure_c: $(SRC_DIR)/pure_c.c include/common.h
	$(BUILD_MONOREPO)
	$(CC) $(CFLAGS) $< $(LIBS) -o $@

$(BIN_DIR)/pthreads_c: $(SRC_DIR)/pthreads_c.c include/common.h
	$(BUILD_MONOREPO)
	$(CC) $(CFLAGS) -pthread $< $(LIBS) -o $@

$(BIN_DIR)/openmp_c: $(SRC_DIR)/openmp_c.c include/common.h
	$(BUILD_MONOREPO)
	$(CC) $(CFLAGS) $(OMP_FLAGS) $< $(LIBS) -o $@

$(BIN_DIR)/metal_c: include/common.h
	$(BUILD_MONOREPO)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -framework Metal -framework Foundation $(SRC_DIR)/metal_c.m $(LIBS) -o $@
else
	$(CC) $(CFLAGS) $(SRC_DIR)/metal_stub.c $(LIBS) -o $@
endif

$(BIN_DIR)/cuda_c: include/common.h
	$(BUILD_MONOREPO)
ifneq ($(NVCC),)
ifeq ($(UNAME_M),aarch64)
	nvcc -O3 -arch=sm_53 -Iinclude $(MONOREPO_INC) $(SRC_DIR)/cuda_c.cu $(MONOREPO_LIB) -o $@
else
	nvcc -O3 -Iinclude $(MONOREPO_INC) $(SRC_DIR)/cuda_c.cu $(MONOREPO_LIB) -o $@
endif
else
	$(CC) $(CFLAGS) $(SRC_DIR)/cuda_stub.c $(LIBS) -o $@
endif

clean:
	rm -rf $(BIN_DIR)

.PHONY: all info clean
