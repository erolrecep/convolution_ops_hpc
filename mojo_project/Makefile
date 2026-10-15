UNAME_S := $(shell uname -s)
CC ?= gcc

ifeq ($(UNAME_S),Darwin)
    DYLIB = src/libc_loader.dylib
    CC = clang
else
    DYLIB = src/libc_loader.so
    CC = gcc
endif

BIN_DIR = bin
SRC_DIR = src
TARGET = $(BIN_DIR)/mojo_conv

all: $(BIN_DIR) $(DYLIB) $(TARGET)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(DYLIB): $(SRC_DIR)/c_loader.c
	$(CC) -O3 -shared -fPIC $< -o $@

PIXI ?= $(shell which pixi 2>/dev/null || echo $(HOME)/.pixi/bin/pixi)

$(TARGET): $(SRC_DIR)/convolution_mojo.mojo $(DYLIB)
	$(PIXI) run mojo build $< -o $@

clean:
	rm -rf $(BIN_DIR) $(SRC_DIR)/*.dylib $(SRC_DIR)/*.so

.PHONY: all clean
