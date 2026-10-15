#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../include/common.h"

__global__ void convolution_kernel_cuda(
    const unsigned char *in, unsigned char *out,
    int width, int height, int channels,
    int x_start, int x_end, int y_start, int y_end)
{
    int x = x_start + blockIdx.x * blockDim.x + threadIdx.x;
    int y = y_start + blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= x_end || y >= y_end) return;

    float kernel[3][3] = {
        {1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f},
        {2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f},
        {1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f}
    };

    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int py = min(max(y + ky, 0), height - 1);
                int px = min(max(x + kx, 0), width - 1);
                sum += (float)in[(py * width + px) * channels + c] * kernel[ky + 1][kx + 1];
            }
        }
        out[(y * width + x) * channels + c] = (unsigned char)(sum < 0.0f ? 0.0f : (sum > 255.0f ? 255.0f : (sum + 0.5f)));
    }
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <technique_num 1-4> <input_image> <output_image>\n", argv[0]);
        return 1;
    }

    int technique = atoi(argv[1]);
    const char *input_path = argv[2];
    const char *output_path = argv[3];

    Image in, out;
    in.data = stbi_load(input_path, &in.width, &in.height, &in.channels, 0);
    if (!in.data) {
        fprintf(stderr, "Failed to load image: %s\n", input_path);
        return 1;
    }

    out.width = in.width;
    out.height = in.height;
    out.channels = in.channels;
    size_t data_size = in.width * in.height * in.channels;
    out.data = (unsigned char *)malloc(data_size);

    unsigned char *d_in, *d_out;
    cudaMalloc(&d_in, data_size);
    cudaMalloc(&d_out, data_size);
    cudaMemcpy(d_in, in.data, data_size, cudaMemcpyHostToDevice);

    int num_regions = 0;
    Region *regions = get_regions(technique, in.width, in.height, &num_regions);

    double start_time = get_time_sec();

    dim3 threadsPerBlock(16, 16);
    for (int i = 0; i < num_regions; i++) {
        int reg_w = regions[i].x_end - regions[i].x_start;
        int reg_h = regions[i].y_end - regions[i].y_start;
        dim3 numBlocks((reg_w + 15) / 16, (reg_h + 15) / 16);

        convolution_kernel_cuda<<<numBlocks, threadsPerBlock>>>(
            d_in, d_out, in.width, in.height, in.channels,
            regions[i].x_start, regions[i].x_end, regions[i].y_start, regions[i].y_end
        );
    }
    cudaDeviceSynchronize();

    double elapsed = get_time_sec() - start_time;

    cudaMemcpy(out.data, d_out, data_size, cudaMemcpyDeviceToHost);
    if (strcmp(output_path, "none") != 0) {
        stbi_write_jpg(output_path, out.width, out.height, out.channels, out.data, 90);
    }

    cudaFree(d_in);
    cudaFree(d_out);
    free(regions);
    stbi_image_free(in.data);
    free(out.data);

    printf("EXECUTION_TIME: %.6f\n", elapsed);
    return 0;
}
