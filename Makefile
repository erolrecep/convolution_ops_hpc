PYTHON = .venv/bin/python3
PIP = .venv/bin/pip

TEST_IMAGE_4K = images/wp6389298-4k-nba-desktop-wallpapers.jpg
BENCHMARK_IMAGE_8K = images/bmw-m3-touring-24h-7680x4320-26531.jpg

all: benchmark

venv:
	@if [ ! -d ".venv" ]; then \
		python3 -m venv .venv; \
		echo "Virtual environment created at .venv"; \
	fi

install: venv
	@$(PIP) install -q --upgrade pip
	@$(PIP) install -q -r requirements.txt
	@echo "Python dependencies installed successfully."

fetch-headers:
	@mkdir -p c_project/include
	@if [ ! -f "c_project/include/stb_image.h" ]; then \
		curl -sSL https://raw.githubusercontent.com/nothings/stb/master/stb_image.h -o c_project/include/stb_image.h; \
	fi
	@if [ ! -f "c_project/include/stb_image_write.h" ]; then \
		curl -sSL https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h -o c_project/include/stb_image_write.h; \
	fi
	@echo "STB headers fetched."

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
	@rm -rf .venv experiments/ benchmark_results.csv *.dat *.gp *.png bar_graph.png line_graph.png scatter_plot.png
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "Project cleaned."

.PHONY: all venv install fetch-headers compile test benchmark update-readme clean
