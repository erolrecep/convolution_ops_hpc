# 8K Convolution Multi-Hardware HPC Benchmark Comparison

**Input Image**: BMW M3 Touring 24h ($7680 \times 4320$, 33.18 Megapixels, 3 RGB channels, 99.5M pixel-channels)  
**Evaluated Platforms**:
1. **Apple M5 Max 16"** (18 CPU Cores, 40 GPU Cores, 36GB Unified RAM, macOS ARM64)
2. **`dlbox`** (AMD Ryzen Threadripper 2950X 16C/32T, NVIDIA GeForce RTX 5060 Ti 16GB, 64GB DDR4, Linux x86_64)
3. **`rpi5ai` (Raspberry Pi 5)** (4× Cortex-A76 @ 2.4 GHz, 8GB LPDDR4X, Debian Linux aarch64)
4. **`rpi01` (Raspberry Pi 4)** (4× Cortex-A72 @ 1.8 GHz, 8GB LPDDR4, Debian Linux aarch64)
5. **`jetson4GB` (NVIDIA Jetson Nano)** (4× Cortex-A57 @ 1.43 GHz, 128-core Maxwell GPU, 4GB LPDDR4, Ubuntu 18.04 aarch64)

---

### Measured Wall-Clock Times Across All Architectures & Languages

| Target / Language | Paradigm & Implementation | Apple M5 Max (18 Cores) | `dlbox` (32 Threads) | Raspberry Pi 5 (4 Cores) | Raspberry Pi 4 (4 Cores) | Jetson Nano (4 Cores) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **C Level 4 Fused Ring-Buffer** | SIMD Intrinsics + OpenMP L1 Cache | **0.0015s (1.5 ms)** | **0.0077s (7.7 ms)** | **0.0334s (33.4 ms)** | **0.0484s (48.4 ms)** | **0.0511s (51.1 ms)** |
| **Level 5 GPU Tiled Kernel** | Metal 4 MSL / CUDA Shared-Mem | **0.00196s (1.96 ms)**| **0.00106s (1.06 ms)**| *N/A* | *N/A* | **0.0662s (66.2 ms)** (0.1315s ZC) |
| **Fortran 2008 (`gfortran`)** | OpenMP `parallel do`, `noalias` | **0.0176s (17.6 ms)** | **0.0674s (67.4 ms)** | **0.2514s (251.4 ms)**| **0.5041s (504.1 ms)**| **0.9621s (962.1 ms)** |
| **Common Lisp (SBCL 2.6.9)** | `sb-thread`, `sb-alien` pointers | **0.0670s (67.0 ms)** | **0.1075s (107.5 ms)**| **2.1175s (2,117 ms)** | **6.2195s (6,220 ms)** | **18.0737s (18,074 ms)**|
| **Go (Go 1.24)** | Goroutines, worker pool | **0.0770s (77.0 ms)** | **0.1410s (141.0 ms)**| **0.9535s (953.5 ms)** | **1.8155s (1,815 ms)** | **3.1913s (3,191 ms)** |
| **Mojo 1.1 (Modular MAX)** | Linear types, heap layout, bit-shifts | **0.1041s (104.1 ms)**| **0.2557s (255.7 ms)**| **0.4753s (475.3 ms)** | *Illegal Inst (ARMv8.0)* | *Illegal Inst (ARMv8.0)* |
| **C with pthreads (Baseline)** | Technique 3 (Row Multi-Thread) | 0.0165s | 0.0724s | 0.2906s | 0.8225s | 1.0514s |
| **C with openMP (Baseline)** | Technique 3 (Row Multi-Thread) | 0.0192s | 0.0814s | 0.3772s | 0.9708s | 1.0850s |
| **Pure C (Single-Core)** | Technique 3 (Sequential Baseline) | 0.1382s | 0.9690s | 1.4947s | 3.3742s | 2.4755s |
| **Python with OpenCV** | Technique 3 (Optimized C++ backend) | 0.0170s | 0.1283s | 0.4800s | 0.8495s | 1.2406s |
| **Python with Numba** | Technique 3 (LLVM JIT) | 0.0278s | 0.1157s | 0.4900s | 0.5164s | N/A |
| **Python with PyTorch** | Technique 3 (Tensor Conv2d MPS/CUDA/CPU)| 0.1994s (MPS) | 0.5390s (CUDA) | 3.4157s | 7.6334s | N/A |
| **Python with NumPy** | Technique 3 (Vectorized slicing) | 0.2563s | 1.1531s | 3.0689s | 10.4802s | 6.5745s |
| **Pure Python** | Technique 1–4 (Interpreted loops) | Timeout (>45s) | Timeout (>45s) | Timeout (>45s) | Timeout (>45s) | Timeout (>45s) |

---

### Peak Hardware Throughput & Speedup Summary

| Device | Fastest Configuration | Best Latency | Throughput (MPixels/s) | Speedup vs 1-Core C |
| :--- | :--- | :---: | :---: | :---: |
| **Apple M5 Max 16"** | C Level 4 Fused Ring-Buffer | **0.0015s (1.5 ms)** | **22,120 MP/s** | **91.3×** |
| **dlbox (Threadripper + RTX 5060 Ti)** | Level 5 CUDA Shared-Memory Kernel | **0.00106s (1.06 ms)**| **31,254 MP/s** | **914.1× (GPU) / 125.8× (CPU)** |
| **Raspberry Pi 5** | C Level 4 Fused Ring-Buffer | **0.0334s (33.4 ms)** | **993.4 MP/s** | **44.8×** |
| **Raspberry Pi 4** | C Level 4 Fused Ring-Buffer | **0.0484s (48.4 ms)** | **685.5 MP/s** | **69.7×** |
| **Jetson Nano 4GB** | C Level 4 Fused Ring-Buffer (CPU) / CUDA (GPU) | **0.0511s (CPU) / 0.0662s (GPU)** | **649.3 MP/s (CPU) / 500.9 MP/s (GPU)** | **48.4× (CPU) / 37.4× (GPU)** |

---

### HPC Architecture & Language Lessons

1. **L1 Cache Locality vs PCIe Bus**:
   The Level 4 Fused Ring-Buffer algorithm keeps intermediate horizontal rows in the L1 data cache. On `dlbox`, this enables the 16-core CPU (**7.7 ms**) to outperform discrete PCIe GPU roundtrip transfer (**58 ms**) by **7.5×**.
2. **Fortran's `noalias` Compiler Advantage**:
   Fortran consistently beat standard C OpenMP 2D across all 5 machines because Fortran language rules guarantee pointer non-aliasing, letting `gfortran -O3` emit unrestricted SIMD vector operations.
3. **Common Lisp (SBCL) Scaling on NUMA vs Edge**:
   SBCL achieved an impressive **15.4× parallel scaling factor** on the 32-thread Threadripper 2950X (**107.5 ms**), but experienced higher dispatch latency on early in-order ARM cores (RPi 4 and Jetson Nano).
4. **ISA Versioning (Modular Mojo)**:
   Mojo 1.1 delivers high performance on modern architectures (**104.1 ms on M5 Max**, **475.3 ms on RPi 5**), but requires ARMv8.2-A+ or newer, producing `SIGILL` on legacy ARMv8.0-A microarchitectures (Cortex-A72 on RPi 4 and Cortex-A57 on Jetson Nano).
