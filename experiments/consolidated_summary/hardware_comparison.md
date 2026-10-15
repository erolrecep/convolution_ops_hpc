# 8K Convolution Multi-Hardware HPC Benchmark Comparison

**Input Image**: BMW M3 Touring 24h (7680x4320, 33.18 Megapixels, 3 channels)

### Measured Wall-Clock Times (seconds)

| Target | Technique | Apple M5 Max 16" (18C / 40GPU / 128GB) | dlbox (Threadripper 2950X + RTX 5060 Ti) | Jetson Nano (4C A57 + Maxwell 128-core) | Raspberry Pi 5 (4C A76 @ 2.4GHz / 16GB) | Raspberry Pi 4 (4C A72 @ 1.5GHz / 8GB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Pure C | Technique 1 | 0.1369s | 1.0422s | 2.5779s | 1.5028s | 3.3717s |
| Pure C | Technique 2 | 0.1420s | 0.9845s | 2.4825s | 1.5005s | 3.3708s |
| Pure C | Technique 3 | 0.1382s | 0.9690s | 2.4755s | 1.4947s | 3.3742s |
| Pure C | Technique 4 | 0.1424s | 1.0153s | 2.5276s | 1.4993s | 3.3647s |
| C with pthreads | Technique 1 | **0.0161s** | **0.0730s** | 0.9957s | 0.2747s | 0.8202s |
| C with pthreads | Technique 2 | 0.1628s | 1.0430s | 2.7452s | 1.0936s | 3.2379s |
| C with pthreads | Technique 3 | **0.0165s** | **0.0724s** | 1.0514s | 0.2906s | 0.8225s |
| C with pthreads | Technique 4 | **0.0163s** | 0.1075s | 1.0665s | 0.2745s | 0.8312s |
| C with openMP | Technique 1 | **0.0169s** | **0.0861s** | 1.0516s | 0.3757s | 0.9686s |
| C with openMP | Technique 2 | 0.1581s | 1.0515s | 2.7513s | 1.4579s | 3.8331s |
| C with openMP | Technique 3 | **0.0192s** | **0.0814s** | 1.0850s | 0.3772s | 0.9708s |
| C with openMP | Technique 4 | **0.0156s** | **0.0903s** | 1.0784s | 0.3759s | 0.9672s |
| C with CUDA | Technique 1 | — | **0.0076s** | 0.1928s | — | — |
| C with CUDA | Technique 2 | — | **0.0077s** | 0.2284s | — | — |
| C with CUDA | Technique 3 | — | **0.0075s** | 0.4439s | — | — |
| C with CUDA | Technique 4 | — | **0.0074s** | 0.2568s | — | — |
| C with Metal | Technique 1 | **0.0110s** | — | — | — | — |
| C with Metal | Technique 2 | **0.0114s** | — | — | — | — |
| C with Metal | Technique 3 | **0.0109s** | — | — | — | — |
| C with Metal | Technique 4 | **0.0117s** | — | — | — | — |
| Pure Python | Technique 1 | *(Timeout >45s)* | *(Timeout >45s)* | *(Timeout >45s)* | *(Timeout >45s)* | *(Timeout >45s)* |
| Pure Python | Technique 2 | *(Timeout >45s)* | *(Timeout >45s)* | *(Timeout >45s)* | *(Timeout >45s)* | *(Timeout >45s)* |
| Pure Python | Technique 3 | *(Timeout >45s)* | *(Timeout >45s)* | *(Timeout >45s)* | *(Timeout >45s)* | *(Timeout >45s)* |
| Pure Python | Technique 4 | *(Timeout >45s)* | *(Timeout >45s)* | *(Timeout >45s)* | *(Timeout >45s)* | *(Timeout >45s)* |
| Python with NumPy | Technique 1 | 0.3911s | 1.3466s | 7.7192s | 3.0127s | 10.5127s |
| Python with NumPy | Technique 2 | 0.3138s | 1.2793s | 8.0965s | 3.1757s | 10.6162s |
| Python with NumPy | Technique 3 | 0.2563s | 1.1531s | 6.5745s | 3.0689s | 10.4802s |
| Python with NumPy | Technique 4 | 0.4760s | 1.6808s | 8.1706s | 3.8602s | 11.5806s |
| Python with OpenCV | Technique 1 | **0.0195s** | 0.1440s | 1.2168s | 0.3882s | 0.8447s |
| Python with OpenCV | Technique 2 | **0.0501s** | **0.0580s** | 1.2476s | 0.4175s | 0.8960s |
| Python with OpenCV | Technique 3 | **0.0170s** | 0.1283s | 1.2406s | 0.4800s | 0.8495s |
| Python with OpenCV | Technique 4 | **0.0193s** | 0.1749s | 1.2031s | 0.4545s | 0.8854s |
| Python with Numba | Technique 1 | **0.0284s** | 0.1048s | — | 0.4548s | 0.5171s |
| Python with Numba | Technique 2 | **0.0216s** | **0.0688s** | — | 0.5528s | 0.5400s |
| Python with Numba | Technique 3 | **0.0278s** | 0.1157s | — | 0.4900s | 0.5164s |
| Python with Numba | Technique 4 | **0.0295s** | 0.1238s | — | 0.5410s | 0.5167s |
| Python with CuPy | Technique 1 | — | — | — | — | — |
| Python with CuPy | Technique 2 | — | — | — | — | — |
| Python with CuPy | Technique 3 | — | — | — | — | — |
| Python with CuPy | Technique 4 | — | — | — | — | — |
| Python with PyTorch | Technique 1 | 0.1967s | 0.6369s | — | 3.2776s | 7.6097s |
| Python with PyTorch | Technique 2 | 0.3949s | 0.6411s | — | 14.2122s | 13.4447s |
| Python with PyTorch | Technique 3 | 0.1994s | 0.5390s | — | 3.4157s | 7.6334s |
| Python with PyTorch | Technique 4 | 0.2007s | 0.6471s | — | 3.4261s | 7.6717s |


### Peak Hardware Throughput & Speedup Summary

| Device | Fastest Target | Best Time (s) | Throughput (MPixels/s) | Speedup vs Single-Thread C |
| :--- | :--- | :---: | :---: | :---: |
| **Apple M5 Max 16" (18C / 40GPU / 128GB)** | C with Metal (Technique 3) | **0.0109s** | **3043.8 MP/s** | **12.56x** |
| **dlbox (Threadripper 2950X + RTX 5060 Ti)** | C with CUDA (Technique 4) | **0.0074s** | **4483.5 MP/s** | **140.84x** |
| **Jetson Nano (4C A57 + Maxwell 128-core)** | C with CUDA (Technique 1) | **0.1928s** | **172.1 MP/s** | **13.37x** |
| **Raspberry Pi 5 (4C A76 @ 2.4GHz / 16GB)** | C with pthreads (Technique 4) | **0.2745s** | **120.9 MP/s** | **5.47x** |
| **Raspberry Pi 4 (4C A72 @ 1.5GHz / 8GB)** | Python with Numba (Technique 3) | **0.5164s** | **64.2 MP/s** | **6.53x** |
