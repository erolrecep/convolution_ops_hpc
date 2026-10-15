#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "../include/common.h"

#define TILE_W 16
#define TILE_H 16
#define SHARED_W 18
#define SHARED_H 18
#define TOTAL_SHARED (SHARED_W * SHARED_H)

__global__ void conv_tiled_separable_cuda(
    const unsigned char * __restrict__ in,
    unsigned char * __restrict__ out,
    int width, int height, int channels)
{
    __shared__ unsigned char s_tile[SHARED_H][SHARED_W][3];

    int base_x = blockIdx.x * TILE_W - 1;
    int base_y = blockIdx.y * TILE_H - 1;
    int linear_tid = threadIdx.y * TILE_W + threadIdx.x;

    // Load first 256 pixels of 18x18 tile (3 channels each)
    if (linear_tid < 256) {
        int sy = linear_tid / SHARED_W;
        int sx = linear_tid % SHARED_W;
        int gy = min(max(base_y + sy, 0), height - 1);
        int gx = min(max(base_x + sx, 0), width - 1);
        int g_idx = (gy * width + gx) * 3;
        s_tile[sy][sx][0] = in[g_idx + 0];
        s_tile[sy][sx][1] = in[g_idx + 1];
        s_tile[sy][sx][2] = in[g_idx + 2];
    }
    // Load remaining 68 pixels
    if (linear_tid < (TOTAL_SHARED - 256)) {
        int idx = 256 + linear_tid;
        int sy = idx / SHARED_W;
        int sx = idx % SHARED_W;
        int gy = min(max(base_y + sy, 0), height - 1);
        int gx = min(max(base_x + sx, 0), width - 1);
        int g_idx = (gy * width + gx) * 3;
        s_tile[sy][sx][0] = in[g_idx + 0];
        s_tile[sy][sx][1] = in[g_idx + 1];
        s_tile[sy][sx][2] = in[g_idx + 2];
    }

    __syncthreads();

    int out_x = blockIdx.x * TILE_W + threadIdx.x;
    int out_y = blockIdx.y * TILE_H + threadIdx.y;

    if (out_x >= width || out_y >= height) return;

    int sy = threadIdx.y + 1;
    int sx = threadIdx.x + 1;

    for (int c = 0; c < 3; c++) {
        // Separable horizontal on 3 adjacent shared memory rows
        int r0 = (int)s_tile[sy - 1][sx - 1][c] + ((int)s_tile[sy - 1][sx][c] << 1) + (int)s_tile[sy - 1][sx + 1][c] + 2;
        r0 >>= 2;

        int r1 = (int)s_tile[sy][sx - 1][c] + ((int)s_tile[sy][sx][c] << 1) + (int)s_tile[sy][sx + 1][c] + 2;
        r1 >>= 2;

        int r2 = (int)s_tile[sy + 1][sx - 1][c] + ((int)s_tile[sy + 1][sx][c] << 1) + (int)s_tile[sy + 1][sx + 1][c] + 2;
        r2 >>= 2;

        // Separable vertical
        int res = (r0 + (r1 << 1) + r2 + 2) >> 2;
        out[(out_y * width + out_x) * 3 + c] = (unsigned char)res;
    }
}

int main(int argc, char **argv) {
    const char *img_path = (argc > 1) ? argv[1] : "images/bmw-m3-touring-24h-7680x4320-26531.jpg";

    Image in;
    in.data = stbi_load(img_path, &in.width, &in.height, &in.channels, 0);
    if (!in.data) {
        fprintf(stderr, "Failed to load image: %s\n", img_path);
        return 1;
    }

    size_t data_size = (size_t)in.width * in.height * in.channels;
    printf("Loaded image: %dx%d (%d channels, %.1f MB)\n",
           in.width, in.height, in.channels, data_size / (1024.0 * 1024.0));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("CUDA Device: %s (Compute %d.%d, Integrated: %s)\n",
           prop.name, prop.major, prop.minor, prop.integrated ? "YES" : "NO");

    unsigned char *d_in, *d_out;
    cudaMalloc(&d_in, data_size);
    cudaMalloc(&d_out, data_size);
    cudaMemcpy(d_in, in.data, data_size, cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_W, TILE_H);
    dim3 gridDim((in.width + TILE_W - 1) / TILE_W, (in.height + TILE_H - 1) / TILE_H);

    // Warmup
    conv_tiled_separable_cuda<<<gridDim, blockDim>>>(d_in, d_out, in.width, in.height, in.channels);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA sync error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Benchmark pure kernel execution time
    int ITERS = 10;
    double total_kernel_time = 0.0;
    for (int i = 0; i < ITERS; i++) {
        double t0 = get_time_sec();
        conv_tiled_separable_cuda<<<gridDim, blockDim>>>(d_in, d_out, in.width, in.height, in.channels);
        cudaDeviceSynchronize();
        total_kernel_time += get_time_sec() - t0;
    }
    double avg_kernel_time = total_kernel_time / ITERS;
    double mp_kernel = ((double)in.width * in.height / 1e6) / avg_kernel_time;

    // Benchmark end-to-end (H2D copy + Kernel + D2H copy)
    double total_e2e_time = 0.0;
    unsigned char *h_out = (unsigned char *)malloc(data_size);
    for (int i = 0; i < ITERS; i++) {
        double t0 = get_time_sec();
        cudaMemcpy(d_in, in.data, data_size, cudaMemcpyHostToDevice);
        conv_tiled_separable_cuda<<<gridDim, blockDim>>>(d_in, d_out, in.width, in.height, in.channels);
        cudaMemcpy(h_out, d_out, data_size, cudaMemcpyDeviceToHost);
        total_e2e_time += get_time_sec() - t0;
    }
    double avg_e2e_time = total_e2e_time / ITERS;
    double mp_e2e = ((double)in.width * in.height / 1e6) / avg_e2e_time;

    printf("\n--- Level 5: CUDA Shared-Memory Tiled Separable Kernel ---\n");
    printf("Pure Kernel Time: %.6fs (%.3f ms) -> %.1f MP/s\n",
           avg_kernel_time, avg_kernel_time * 1000.0, mp_kernel);
    printf("End-to-End Time:  %.6fs (%.3f ms) -> %.1f MP/s\n",
           avg_e2e_time, avg_e2e_time * 1000.0, mp_e2e);

    if (prop.integrated) {
        unsigned char *h_in_zero, *h_out_zero;
        cudaHostAlloc((void**)&h_in_zero, data_size, cudaHostAllocMapped);
        cudaHostAlloc((void**)&h_out_zero, data_size, cudaHostAllocMapped);
        memcpy(h_in_zero, in.data, data_size);

        unsigned char *d_in_zero, *d_out_zero;
        cudaHostGetDevicePointer((void**)&d_in_zero, h_in_zero, 0);
        cudaHostGetDevicePointer((void**)&d_out_zero, h_out_zero, 0);

        // Warmup
        conv_tiled_separable_cuda<<<gridDim, blockDim>>>(d_in_zero, d_out_zero, in.width, in.height, in.channels);
        cudaDeviceSynchronize();

        double total_zc_time = 0.0;
        for (int i = 0; i < ITERS; i++) {
            double t0 = get_time_sec();
            conv_tiled_separable_cuda<<<gridDim, blockDim>>>(d_in_zero, d_out_zero, in.width, in.height, in.channels);
            cudaDeviceSynchronize();
            total_zc_time += get_time_sec() - t0;
        }
        double avg_zc_time = total_zc_time / ITERS;
        double mp_zc = ((double)in.width * in.height / 1e6) / avg_zc_time;
        printf("Zero-Copy Mapped Time: %.6fs (%.3f ms) -> %.1f MP/s (No cudaMemcpy overhead!)\n",
               avg_zc_time, avg_zc_time * 1000.0, mp_zc);

        cudaFreeHost(h_in_zero);
        cudaFreeHost(h_out_zero);
    }

    // Mathematical verification vs Ground Truth
    #define CLAMP_V(v, l, h) (((v) < (l)) ? (l) : (((v) > (h)) ? (h) : (v)))
    int mismatches = 0;
    int max_diff = 0;
    for (int y = 0; y < in.height; y++) {
        for (int x = 0; x < in.width; x++) {
            for (int c = 0; c < in.channels; c++) {
                int p00 = in.data[(CLAMP_V(y-1, 0, in.height-1)*in.width + CLAMP_V(x-1, 0, in.width-1))*3 + c];
                int p01 = in.data[(CLAMP_V(y-1, 0, in.height-1)*in.width + CLAMP_V(x,   0, in.width-1))*3 + c];
                int p02 = in.data[(CLAMP_V(y-1, 0, in.height-1)*in.width + CLAMP_V(x+1, 0, in.width-1))*3 + c];
                int p10 = in.data[(CLAMP_V(y,   0, in.height-1)*in.width + CLAMP_V(x-1, 0, in.width-1))*3 + c];
                int p11 = in.data[(CLAMP_V(y,   0, in.height-1)*in.width + CLAMP_V(x,   0, in.width-1))*3 + c];
                int p12 = in.data[(CLAMP_V(y,   0, in.height-1)*in.width + CLAMP_V(x+1, 0, in.width-1))*3 + c];
                int p20 = in.data[(CLAMP_V(y+1, 0, in.height-1)*in.width + CLAMP_V(x-1, 0, in.width-1))*3 + c];
                int p21 = in.data[(CLAMP_V(y+1, 0, in.height-1)*in.width + CLAMP_V(x,   0, in.width-1))*3 + c];
                int p22 = in.data[(CLAMP_V(y+1, 0, in.height-1)*in.width + CLAMP_V(x+1, 0, in.width-1))*3 + c];

                int r0 = (p00 + (p01 << 1) + p02 + 2) >> 2;
                int r1 = (p10 + (p11 << 1) + p12 + 2) >> 2;
                int r2 = (p20 + (p21 << 1) + p22 + 2) >> 2;
                int expected = (r0 + (r1 << 1) + r2 + 2) >> 2;

                int actual = h_out[(y * in.width + x) * 3 + c];
                int diff = abs(actual - expected);
                if (diff > max_diff) max_diff = diff;
                if (diff > 0) mismatches++;
            }
        }
    }
    printf("Mathematical Verification vs Separable Integer Ground Truth:\n");
    printf("  Total Pixel-Channels: %zu\n", data_size);
    printf("  Max Absolute Difference: %d\n", max_diff);
    printf("  Bit-Exact Matches:       %zu / %zu (%.4f%%)\n",
           data_size - mismatches, data_size, 100.0 * (data_size - mismatches) / data_size);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_out);
    stbi_image_free(in.data);
    return 0;
}
