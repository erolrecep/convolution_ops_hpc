PYTHON = .venv/bin/python3
PIP = .venv/bin/pip

TEST_IMAGE_4K = images/wp6389298-4k-nba-desktop-wallpapers.jpg
BENCHMARK_IMAGE_8K = images/bmw-m3-touring-24h-7680x4320-26531.jpg

all: help

help:
	@echo "========================================================================="
	@echo "🖼️  Convolution Operations HPC Benchmark — Makefile Commands"
	@echo "========================================================================="
	@echo " Setup & Build:"
	@echo "   make sys-deps      - Check and auto-install system software (gnuplot, libomp, curl)"
	@echo "   make install       - Set up Python .venv and install dependencies"
	@echo "   make fetch-headers - Download STB image processing C headers"
	@echo "   make compile       - Compile all C, OpenMP, pthreads, Metal, and CUDA binaries"
	@echo ""
	@echo " Execution & Benchmarks:"
	@echo "   make test          - Run quick 4K verification test across all techniques"
	@echo "   make benchmark     - Run full 8K benchmark suite & generate gnuplot graphs"
	@echo "   make update-readme - Run benchmark and refresh the timing table in README.md"
	@echo ""
	@echo " Cleanup:"
	@echo "   make clean         - Remove compiled binaries, .venv, plots, and experiment data"
	@echo "========================================================================="

sys-deps:
	@echo "--- Checking System Software Dependencies ---"
	@UNAME_S=$$(uname -s); \
	MISSING_DEPS=""; \
	which gnuplot >/dev/null 2>&1 || MISSING_DEPS="$$MISSING_DEPS gnuplot"; \
	which curl >/dev/null 2>&1 || MISSING_DEPS="$$MISSING_DEPS curl"; \
	if [ "$$UNAME_S" = "Darwin" ]; then \
		if [ ! -d "/opt/homebrew/opt/libomp" ] && [ ! -d "/usr/local/opt/libomp" ]; then \
			MISSING_DEPS="$$MISSING_DEPS libomp"; \
		fi; \
	fi; \
	if [ -n "$$MISSING_DEPS" ]; then \
		echo "Missing system software detected:$$MISSING_DEPS"; \
		if [ "$$UNAME_S" = "Darwin" ] && which brew >/dev/null 2>&1; then \
			echo "Installing missing system software using Homebrew..."; \
			brew install $$MISSING_DEPS || true; \
		elif which apt-get >/dev/null 2>&1; then \
			echo "Installing missing system software using apt-get..."; \
			sudo apt-get update && sudo apt-get install -y $$MISSING_DEPS || true; \
		elif which dnf >/dev/null 2>&1; then \
			echo "Installing missing system software using dnf..."; \
			sudo dnf install -y $$MISSING_DEPS || true; \
		else \
			echo "Warning: Package manager not found. Please manually install:$$MISSING_DEPS"; \
		fi; \
	else \
		echo "All required system software dependencies are installed."; \
	fi

venv: sys-deps
	@if [ ! -d ".venv" ]; then \
		python3 -m venv .venv; \
		echo "Virtual environment created at .venv"; \
	fi

install: venv
	@$(PIP) install -q --upgrade pip
	@$(PIP) install -q -r requirements.txt
	@echo "Python dependencies installed successfully."

fetch-headers:
	@mkdir -p "../../_c_/include"
	@if [ ! -f "../../_c_/include/stb_image.h" ]; then \
		curl -sSL https://raw.githubusercontent.com/nothings/stb/master/stb_image.h -o "../../_c_/include/stb_image.h"; \
	fi
	@if [ ! -f "../../_c_/include/stb_image_write.h" ]; then \
		curl -sSL https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h -o "../../_c_/include/stb_image_write.h"; \
	fi
	@echo "STB headers verified in monorepo _c_/include."

compile: fetch-headers
	@$(MAKE) -C c_project

test: install compile
	@echo "Running quick 4K functionality test..."
	@$(PYTHON) scripts/benchmark_runner.py $(TEST_IMAGE_4K)

benchmark: install compile
	@echo "Running full 8K benchmark suite..."
	@$(PYTHON) scripts/benchmark_runner.py $(BENCHMARK_IMAGE_8K)

update-readme: benchmark

clean:
	@$(MAKE) -C c_project clean
	@rm -rf *.dat *.gp *.png bar_graph.png line_graph.png scatter_plot.png
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "Project cleaned (experiments preserved)."

.PHONY: all help sys-deps venv install fetch-headers compile test benchmark update-readme clean
