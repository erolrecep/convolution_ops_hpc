FC ?= gfortran
CC ?= gcc

FFLAGS = -O3 -fopenmp -Wall
CFLAGS = -O3

BIN_DIR = bin
SRC_DIR = src

TARGET = $(BIN_DIR)/fortran_conv

all: $(BIN_DIR) $(TARGET)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/c_bridge.o: $(SRC_DIR)/c_bridge.c
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET): $(SRC_DIR)/convolution_fortran.f90 $(BIN_DIR)/c_bridge.o
	$(FC) $(FFLAGS) $^ -o $@ -lm

clean:
	rm -rf $(BIN_DIR)

.PHONY: all clean
