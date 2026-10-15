# High Performance Computing: 8K Image Convolution Benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Platforms: macOS | Linux | ARM64 | x86_64](https://img.shields.io/badge/Platforms-macOS%20|%20Linux%20|%20ARM64%20|%20x86__64-brightgreen.svg)]()
[![Languages: C | Fortran | Go | Mojo | Lisp | Python | CUDA | Metal](https://img.shields.io/badge/Languages-C%20|%20Fortran%20|%20Go%20|%20Mojo%20|%20Lisp%20|%20Python%20|%20CUDA%20|%20Metal-orange.svg)]()

A comprehensive High Performance Computing (HPC) benchmarking suite evaluating 2D stencil image convolution ($3 \times 3$ Gaussian Blur) on an **8K Ultra-HD image** ($7680 \times 4320$, 3 RGB channels, **99,532,800 pixel-channel evaluations** per run).

The benchmark rigorously measures execution time, memory locality, and hardware scaling across **5 physical hardware architectures** using **8 programming languages** and multiple parallel programming models.

---

## 1. Physical Hardware Evaluation Platforms

| Platform ID | System Architecture | CPU Microarchitecture | Cores / Threads | GPU & Accelerator | RAM / Memory Subsystem |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **Apple M5 Max 16"** | Apple Silicon ARM64 | M5 Max (Performance & Efficiency) | 18 Cores | 40-Core Metal 4 GPU | 36 GB Unified Memory (UMA, >400 GB/s) |
| **`dlbox`** | Workstation x86_64 | AMD Ryzen Threadripper 2950X | 16 Cores / 32 Threads | NVIDIA GeForce RTX 5060 Ti 16GB | 64 GB Quad-Channel DDR4 (NUMA) |
| **`rpi5ai` (Raspberry Pi 5)**| Single Board Computer ARM64 | Broadcom BCM2712 (Cortex-A76 @ 2.4 GHz) | 4 Cores | VideoCore VII | 8 GB LPDDR4X-4267 |
| **`rpi01` (Raspberry Pi 4)** | Single Board Computer ARM64 | Broadcom BCM2711 (Cortex-A72 @ 1.8 GHz) | 4 Cores | VideoCore VI | 8 GB LPDDR4-3200 |
| **`jetson4GB` (Jetson Nano)** | Embedded AI SoC ARM64 | NVIDIA Tegra X1 (Cortex-A57 @ 1.43 GHz) | 4 Cores | 128-Core Maxwell GPU | 4 GB 64-bit LPDDR4 (Shared SoC RAM) |

---

## 2. Comprehensive 8K Benchmark Results

All tests evaluate a $7680 \times 4320$ 24-bit RGB image (`images/bmw-m3-touring-24h-7680x4320-26531.jpg`). Times reported are pure wall-clock filter computation times in seconds (best of 3 runs, Technique 3 / Row Partitioning).

| Technology / Implementation | Paradigm & Optimization Tier | Apple M5 Max (18 Cores) | `dlbox` (32 Threads) | Raspberry Pi 5 (4 Cores) | Raspberry Pi 4 (4 Cores) | Jetson Nano (4 Cores) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **C Level 4 Fused Ring-Buffer** | SIMD Intrinsics + OpenMP L1 Cache Reuse | **0.0015s (1.5 ms)** | **0.0077s (7.7 ms)** | **0.0334s (33.4 ms)** | **0.0484s (48.4 ms)** | **0.0511s (51.1 ms)** |
| **Level 5 GPU Tiled Kernel** | Metal 4 MSL / CUDA Shared-Mem Tile | **0.00196s (1.96 ms)**| **0.00106s (1.06 ms)**| *N/A* | *N/A* | **0.0662s (66.2 ms)** (0.1315s ZC) |
| **Fortran 2008 (`gfortran`)** | OpenMP `parallel do`, `noalias` guarantees | **0.0176s (17.6 ms)** | **0.0674s (67.4 ms)** | **0.2514s (251.4 ms)**| **0.5041s (504.1 ms)**| **0.9621s (962.1 ms)** |
| **Common Lisp (SBCL 2.6.9)** | `sb-thread`, `sb-alien` raw pointers | **0.0670s (67.0 ms)** | **0.1075s (107.5 ms)**| **2.1175s (2,118 ms)** | **6.2195s (6,220 ms)** | **18.0737s (18,074 ms)**|
| **Go (Go 1.24)** | Goroutines, strip partition worker pool | **0.0770s (77.0 ms)** | **0.1410s (141.0 ms)**| **0.9535s (953.5 ms)** | **1.8155s (1,815 ms)** | **3.1913s (3,191 ms)** |
| **Mojo 1.1 (Modular MAX)** | Linear types, heap layout, bit-shifts | **0.1041s (104.1 ms)**| **0.2557s (255.7 ms)**| **0.4753s (475.3 ms)** | *Illegal Inst (ARMv8.0)* | *Illegal Inst (ARMv8.0)* |
| **C with pthreads (Baseline)** | Multi-threaded POSIX threads 2D stencil | 0.0165s (16.5 ms) | 0.0724s (72.4 ms) | 0.2906s (290.6 ms) | 0.8225s (822.5 ms) | 1.0514s (1,051 ms) |
| **C with openMP (Baseline)** | Multi-threaded OpenMP 2D stencil | 0.0192s (19.2 ms) | 0.0814s (81.4 ms) | 0.3772s (377.2 ms) | 0.9708s (970.8 ms) | 1.0850s (1,085 ms) |
| **Pure C (Baseline 1-Core)** | Scalar sequential loop | 0.1382s (138.2 ms) | 0.9690s (969.0 ms) | 1.4947s (1,495 ms) | 3.3742s (3,374 ms) | 2.4755s (2,475 ms) |
| **Python with OpenCV** | Multi-threaded C++ backend (`cv2.filter2D`) | 0.0170s (17.0 ms) | 0.1283s (128.3 ms) | 0.4800s (480.0 ms) | 0.8495s (849.5 ms) | 1.2406s (1,241 ms) |
| **Python with Numba** | LLVM JIT compiled multi-threading | 0.0278s (27.8 ms) | 0.1157s (115.7 ms) | 0.4900s (490.0 ms) | 0.5164s (516.4 ms) | *N/A (Python 3.6)* |
| **Python with PyTorch** | Tensor Conv2d (MPS / CUDA / CPU) | 0.1994s (MPS) | 0.5390s (CUDA) | 3.4157s (CPU) | 7.6334s (CPU) | *N/A* |
| **Python with NumPy** | Vectorized slicing array expressions | 0.2563s (256.3 ms) | 1.1531s (1,153 ms) | 3.0689s (3,069 ms) | 10.4802s (10,480 ms)| 6.5745s (6,575 ms) |
| **Pure Python** | Triple-nested interpreted loops | *Timeout (>45s)* | *Timeout (>45s)* | *Timeout (>45s)* | *Timeout (>45s)* | *Timeout (>45s)* |

---

## 3. Domain Decomposition Techniques

The benchmark tests 4 distinct spatial decomposition patterns across all targets:

```
Technique 1: Equal Grid (Core count N)      Technique 2: 9 Regions (Load Imbalance Benchmark)
+-------------+-------------+                +-----+-------------------+-----+
|             |             |                |  1  |         2         |  3  |
|      1      |      2      |                +-----+-------------------+-----+
|             |             |                |     |                   |     |
+-------------+-------------+                |  4  |         5         |  6  |
|             |             |                |     | (99.9% of pixels) |     |
|      3      |      4      |                +-----+-------------------+-----+
|             |             |                |  7  |         8         |  9  |
+-------------+-------------+                +-----+-------------------+-----+

Technique 3: Row Slices (Cache Continuous)  Technique 4: Column Slices (Strided Jumps)
+---------------------------+                +---+---+---+---+---+---+---+---+
|             1             |                |   |   |   |   |   |   |   |   |
+---------------------------+                |   |   |   |   |   |   |   |   |
|             2             |                | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
+---------------------------+                |   |   |   |   |   |   |   |   |
|             3             |                |   |   |   |   |   |   |   |   |
+---------------------------+                +---+---+---+---+---+---+---+---+
```

---

## 4. Hardware Installation & Setup Instructions

Below are exact, verified setup commands tailored for each supported operating system and architecture:

### A. macOS (Apple Silicon M-Series)
```bash
# 1. Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install compilers and tools
brew install libomp gnuplot curl sbcl gcc go

# 3. Install Pixi package manager for Modular Mojo
curl -fsSL https://pixi.sh/install.sh | bash
source ~/.zshrc

# 4. Clone repository
git clone https://github.com/erolrecep/convolution_ops_hpc.git
cd convolution_ops_hpc

# 5. Build all targets
make sys-deps
make compile
make -C fortran_project
make -C lisp_project
make -C mojo_project
```

### B. Linux x86_64 (`dlbox` / Ubuntu / Debian with NVIDIA GPU)
```bash
# 1. Install base development tools and compilers
sudo apt-get update
sudo apt-get install -y build-essential gfortran gnuplot curl git libomp-dev

# 2. Install Pixi (manages SBCL and Mojo on Linux x86_64)
curl -fsSL https://pixi.sh/install.sh | bash
source ~/.bashrc

# 3. Install SBCL and Go
pixi global install sbcl --channel conda-forge
# Or via apt: sudo apt-get install -y sbcl golang-go

# 4. Install NVIDIA CUDA Toolkit (if not already installed)
sudo apt-get install -y nvidia-cuda-toolkit

# 5. Build all targets
cd ~/Developer/codebase/apps/convolution_ops_hpc
make compile
make -C fortran_project
make -C lisp_project
make -C mojo_project
```

### C. Raspberry Pi 5 (`rpi5ai` / Debian aarch64)
```bash
# 1. Install native compilers and runtimes
sudo apt-get update
sudo apt-get install -y build-essential gfortran sbcl golang-go gnuplot curl

# 2. Install Pixi (for Modular Mojo on linux-aarch64)
curl -fsSL https://pixi.sh/install.sh | bash
source ~/.bashrc

# 3. Build project
cd convolution_ops_hpc
make compile
make -C fortran_project
make -C lisp_project
make -C mojo_project
```

### D. Raspberry Pi 4 (`rpi01` / Debian Bookworm aarch64)
```bash
# 1. Install native compilers
sudo apt-get update
sudo apt-get install -y build-essential gfortran sbcl golang-go gnuplot curl

# 2. Build project
cd convolution_ops_hpc
make compile
make -C fortran_project
make -C lisp_project
# Note: Mojo requires ARMv8.2-A+ instructions; on Cortex-A72 (ARMv8.0), use Fortran/C/Lisp/Go.
```

### E. NVIDIA Jetson Nano (`jetson4GB` / JetPack L4T 32.7 Ubuntu 18.04)
```bash
# 1. Install development packages
sudo apt-get update
sudo apt-get install -y build-essential gfortran sbcl gnuplot curl

# 2. Verify CUDA PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 3. Build project
cd convolution_ops_hpc
make compile
make -C fortran_project
make -C lisp_project
```

---

## 5. Execution Commands: Running Each Technology

Every language implementation supports the universal CLI contract:
```bash
<executable> <technique 1-4> <input_image_path> <output_image_path | "none">
```
Passing `"none"` as the output path skips disk I/O to measure pure in-memory computation latency.

### 1. Unified Automated Benchmark
```bash
# Runs full benchmark suite across all available targets on the current machine
make benchmark

# Run a quick 4K verification test
make test

# Display all Makefile commands
make help
```

### 2. Running Individual Languages Directly

#### A. C (Pure, OpenMP, Pthreads, SIMD, Fused Ring-Buffer)
```bash
# Compile C suite
make -C c_project

# Run Level 4 Fused Ring-Buffer (Fastest CPU implementation)
./c_project/bin/fused_ringbuffer_c 3 images/bmw-m3-touring-24h-7680x4320-26531.jpg none

# Run OpenMP 2D baseline
./c_project/bin/openmp_c 3 images/bmw-m3-touring-24h-7680x4320-26531.jpg none
```

#### B. GPU Accelerators (Metal 4 & NVIDIA CUDA)
```bash
# Apple Silicon Metal (macOS)
./c_project/bin/metal_c 3 images/bmw-m3-touring-24h-7680x4320-26531.jpg none
# Level 5 Tiled Separable Metal Shader:
./c_project/bin/test_metal_separable 3 images/bmw-m3-touring-24h-7680x4320-26531.jpg none

# NVIDIA CUDA (Linux / Jetson)
./c_project/bin/cuda_c 3 images/bmw-m3-touring-24h-7680x4320-26531.jpg none
# Level 5 Shared-Memory Tiled CUDA Kernel:
./c_project/bin/test_cuda_separable 3 images/bmw-m3-touring-24h-7680x4320-26531.jpg none
```

#### C. Modern Fortran 2008 (OpenMP)
```bash
# Compile Fortran binary
make -C fortran_project

# Run Fortran 8K convolution
./fortran_project/bin/fortran_conv 3 images/bmw-m3-touring-24h-7680x4320-26531.jpg none
```

#### D. Common Lisp (SBCL Native POSIX Threads)
```bash
# Compile standalone SBCL executable
make -C lisp_project

# Run Common Lisp 8K convolution
./lisp_project/bin/lisp_conv 3 images/bmw-m3-touring-24h-7680x4320-26531.jpg none
```

#### E. Go (Golang Concurrency)
```bash
# Run precompiled Go binary (Darwin ARM64 / Linux AMD64 / Linux ARM64)
./go_project/bin/go_conv 3 images/bmw-m3-touring-24h-7680x4320-26531.jpg none

# Or compile locally
cd go_project && go build -o bin/go_conv main.go
```

#### F. Mojo 1.1 (Modular MAX)
```bash
# Compile standalone Mojo executable via Pixi
make -C mojo_project

# Run Mojo 8K convolution
./mojo_project/bin/mojo_conv 3 images/bmw-m3-touring-24h-7680x4320-26531.jpg none
```

#### G. Python Suite (NumPy, OpenCV, Numba, PyTorch)
```bash
# Activate virtual environment
source .venv/bin/activate

# Run full Python benchmarks
python3 scripts/benchmark_runner.py images/bmw-m3-touring-24h-7680x4320-26531.jpg
```

---

## 6. Key HPC & Computer Architecture Lessons

1. **Cache Locality Trumps Raw Peak FLOPs**:
   The Level 4 Fused Ring-Buffer algorithm eliminates main memory round-trips by streaming intermediate horizontal rows through an L1-cached circular buffer. On `dlbox`, this allows the CPU (**7.7 ms**) to outperform discrete PCIe GPU roundtrip transfer (**58 ms**) by **7.5×**.
2. **Strict Compiler Aliasing Guarantees (`noalias`)**:
   Fortran consistently beat standard C OpenMP across all physical CPUs (17.6 ms on M5 Max, 67.4 ms on Threadripper, 251.4 ms on RPi 5, 504.1 ms on RPi 4, 962.1 ms on Jetson Nano). Because Fortran guarantees that arrays do not alias, `gfortran -O3` emits unrestricted SIMD vector operations without defensive scalar checks.
3. **Common Lisp (SBCL) Scaling on NUMA vs Embedded**:
   SBCL achieved an exceptional **15.4× parallel scaling factor** on the 32-thread Threadripper 2950X (**107.5 ms**), but experienced higher dispatch latency on early in-order ARM cores (RPi 4 and Jetson Nano).
4. **Instruction Set Versioning in Modern AI Compilers**:
   Modular Mojo compiles targeting **ARMv8.2-A+**, executing with high efficiency on modern nodes (475.3 ms on Raspberry Pi 5 Cortex-A76), while producing `SIGILL` on legacy ARMv8.0-A architectures (Raspberry Pi 4 and Jetson Nano).
5. **Unified Memory Architecture (UMA) Eliminates Bus Latency**:
   On Apple Silicon (M5 Max), the CPU, GPU, and Neural Engine share a single physical pool of memory exceeding 400 GB/s bandwidth. Metal compute shaders process image data directly at memory bus speeds with zero copy overhead, achieving **1.96 ms** latency and **16,953 Megapixels/sec**.

---

## 7. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.