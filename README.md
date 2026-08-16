# Convolution Operation with High Performance Computing

In this project, I am going to apply convolution operation (image filtering at general) to a single 8K image using different programming techniques to benchmark them against each other.

Makefile will be the control center of the project, it will be used to compile and run the different techniques and benchmark all of them then write the results to a csv file. Using gnuplot, Makefile will plot the benchmark results as a bar graph, line graph, and scatter plot to visualize the performance difference between the different techniques.

## Image Processing with C Programs
Using `stb_image.h` and `stb_image_write.h`, I will read the 8k image and write the output image.

## Techniques
 - Pure C
 - C with pthreads
 - C with openMP
 - C with CUDA
 - C with Apple Silicon Metal Framework
 - Pure Python
 - Python with NumPy
 - Python with OpenCV
 - Python with Numba
 - Python with CuPy
 - Pyhon with PyTorch

## Image Processing Techniques

### Technique 1

Split image into equal n regions based on the number of cores available on the system. Then apply convolution on each region.

```bash
----------------------------
|            |             |
|      1     |      2      |
|            |             |
----------------------------
|            |             |
|      3     |      4      |
|            |             |
----------------------------
```

### Technique 2

Split image into 9 regions and use thread/core and apply convolution on each region.

```bash
---------------------------------
| 1 |           2           | 3 |
---------------------------------
|   |                       |   |
| 4 |           5           | 6 |
|   |                       |   |
---------------------------------
| 7 |           8           | 9 |
---------------------------------
```

### Technique 3

Split image row based into n regions then apply convolution on each region.

```bash
-------------------------------------------------
|                                               |
|                       1                       |
|                                               |
-------------------------------------------------
|                                               |
|                       2                       |
|                                               |
-------------------------------------------------
|                                               |
|                       3                       |
|                                               |
-------------------------------------------------
|                                               |
|                       4                       |
|                                               |
-------------------------------------------------
```

### Technique 4

Split image into n equal columns based on the number of cores available on the system. Then apply convolution on each column.

```bash
-------------------------------------------------
|       |       |       |       |       |       |
|       |       |       |       |       |       |
|       |       |       |       |       |       |
|   1   |   2   |   3   |   4   |   5   |   6   |
|       |       |       |       |       |       |
|       |       |       |       |       |       |
|       |       |       |       |       |       |
-------------------------------------------------
```


## Expected Benchmark Table

Build a 2D table on one-side is programming language with tools and other size is the programming technique. 

 - Rows are the programming languages with tools.
 - Columns are the programming techniques.

| |Technique 1|Technique 2|Technique 3|Technique 4|
|--|--|--|--|--|
|**Pure C**| 0.2145s | 0.2091s | 0.2082s | 0.2192s |
|**C with pthreads**| 0.0295s | 0.2144s | 0.0277s | 0.0303s |
|**C with openMP**| 0.0277s | 0.2231s | 0.0274s | 0.0302s |
|**C with CUDA**| N/A | N/A | N/A | N/A |
|**C with Metal**| 0.0118s | 0.0106s | 0.0109s | 0.0103s |
|**Pure Python**| Timeout (>45s) | Timeout (>45s) | Timeout (>45s) | Timeout (>45s) |
|**Python with NumPy**| 0.7173s | 0.6619s | 0.4621s | 0.7502s |
|**Python with OpenCV**| 0.0340s | 0.0508s | 0.0297s | 0.0279s |
|**Python with Numba**| 0.3996s | 0.3905s | 0.3978s | 0.3955s |
|**Python with CuPy**| N/A | N/A | N/A | N/A |
|**Python with PyTorch**| 0.2711s | 0.3989s | 0.2623s | 0.2574s |


## Getting Started & Execution Instructions

This project is built to compile and run on any macOS, Linux, or Windows (WSL) machine with automatic environment and hardware compiler detection.

### 1. Prerequisites
Ensure the following tools are installed on your system:
- **C Compiler**: `gcc` or `clang`
- **Make**: GNU Make build utility
- **Python**: Python 3.8+ (with `venv` module)
- **Gnuplot**: Required for automated graph plotting
- **OpenMP Library** *(Optional)*: E.g., `libomp` on macOS (`brew install libomp`) or `libgomp` on Linux
- **CUDA Toolkit** *(Optional)*: `nvcc` compiler (if running on a machine with an NVIDIA GPU for CUDA/CuPy execution)

### 2. Project Directory Setup
- Save your test and benchmark images inside the [`images/`](file:///Users/erolrecep/Downloads/convolution_ops_hpc/images) folder.
- Configure image targets in the top-level [`Makefile`](file:///Users/erolrecep/Downloads/convolution_ops_hpc/Makefile):
  ```makefile
  TEST_IMAGE_4K = images/your_test_image.jpg
  BENCHMARK_IMAGE_8K = images/your_benchmark_image.jpg
  ```

### 3. Execution Commands
All actions are orchestrated via the main Makefile in the root directory:

* **Compile Code**: Detects operating system and GPU compilers (like `nvcc`) to build support stubs or full hardware-accelerated binaries:
  ```bash
  make compile
  ```
* **Run Functionality Test (4K Image)**: Set up python virtual environment, installs dependencies, builds C binaries, and runs a quick verification test:
  ```bash
  make test
  ```
* **Run Full Benchmark (8K Image)**: Compiles all targets, runs 4 techniques across the 11 target configurations, updates the timing table in `README.md`, and runs gnuplot pipelines:
  ```bash
  make benchmark
  ```
  *Or simply execute the default target:*
  ```bash
  make
  ```
* **Clean the Workspace**: Clears out compiled binaries, Python `.venv` virtual environment, experiment reports, and bytecode caches:
  ```bash
  make clean
  ```

### 4. Benchmark Outputs
Each execution of `make benchmark` generates an isolated experiment run folder:
- **Location**: `experiments/<YYYYMMDD_HHMMSS>/`
- **Content**:
  - `experiment.yaml`: Detailed metadata of the test run, total run counts, and active order of experiments.
  - `benchmark_results.csv`: Table containing the timing calculations for each target and technique.
  - `results/`: Contains generated performance graphs:
    - `bar_graph.png`
    - `line_graph.png`
    - `scatter_plot.png`
 