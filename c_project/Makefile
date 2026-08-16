CC = clang
CFLAGS = -O3 -Iinclude
OMP_FLAGS = -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -Xpreprocessor -fopenmp -lomp

BIN_DIR = bin
SRC_DIR = src

# Dynamic Hardware Detection
NVCC := $(shell which nvcc 2>/dev/null)
UNAME_S := $(shell uname -s)

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
	$(CC) $(CFLAGS) $< -o $@

$(BIN_DIR)/pthreads_c: $(SRC_DIR)/pthreads_c.c include/common.h
	$(CC) $(CFLAGS) -pthread $< -o $@

$(BIN_DIR)/openmp_c: $(SRC_DIR)/openmp_c.c include/common.h
	$(CC) $(CFLAGS) $(OMP_FLAGS) $< -o $@

$(BIN_DIR)/metal_c: include/common.h
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -framework Metal -framework Foundation $(SRC_DIR)/metal_c.m -o $@
else
	$(CC) $(CFLAGS) $(SRC_DIR)/metal_stub.c -o $@
endif

$(BIN_DIR)/cuda_c: include/common.h
ifneq ($(NVCC),)
	nvcc -O3 -Iinclude $(SRC_DIR)/cuda_c.cu -o $@
else
	$(CC) $(CFLAGS) $(SRC_DIR)/cuda_stub.c -o $@
endif

clean:
	rm -rf $(BIN_DIR)

.PHONY: all info clean
