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
TARGET = $(BIN_DIR)/lisp_conv

SBCL ?= $(shell which sbcl 2>/dev/null || echo $(HOME)/.pixi/bin/sbcl)

all: $(BIN_DIR) $(DYLIB) $(TARGET)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(DYLIB): $(SRC_DIR)/c_loader.c
	$(CC) -O3 -shared -fPIC $< -o $@

$(TARGET): $(SRC_DIR)/convolution_lisp.lisp $(DYLIB)
	$(SBCL) --noinform --disable-debugger --load $(SRC_DIR)/convolution_lisp.lisp \
		--eval '(sb-ext:save-lisp-and-die "$@" :toplevel #'\''main :executable t)'

clean:
	rm -rf $(BIN_DIR) $(SRC_DIR)/*.dylib $(SRC_DIR)/*.so

.PHONY: all clean
