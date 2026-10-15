#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "../include/common.h"
#include "../include/separable.h"

// Multithreaded OpenMP Separable Convolution (Floating Point)
void conv_separable_omp_float(const Image *in, Image *out, float *temp) {
    int W = in->width;
    int H = in->height;
    int ch = in->channels;

    // Pass 1: Horizontal Blur parallelized over rows
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; y++) {
        int row_idx = y * W;
        for (int x = 0; x < W; x++) {
            int x_prev = (x > 0) ? x - 1 : 0;
            int x_next = (x < W - 1) ? x + 1 : W - 1;
            for (int c = 0; c < ch; c++) {
                float v_prev = in->data[(row_idx + x_prev) * ch + c];
                float v_curr = in->data[(row_idx + x) * ch + c];
                float v_next = in->data[(row_idx + x_next) * ch + c];
                temp[(row_idx + x) * ch + c] = 0.25f * v_prev + 0.5f * v_curr + 0.25f * v_next;
            }
        }
    }

    // Pass 2: Vertical Blur parallelized over rows
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; y++) {
        int y_prev = (y > 0) ? y - 1 : 0;
        int y_next = (y < H - 1) ? y + 1 : H - 1;
        int row_prev = y_prev * W;
        int row_curr = y * W;
        int row_next = y_next * W;

        for (int x = 0; x < W; x++) {
            for (int c = 0; c < ch; c++) {
                float v_prev = temp[(row_prev + x) * ch + c];
                float v_curr = temp[(row_curr + x) * ch + c];
                float v_next = temp[(row_next + x) * ch + c];
                float sum = 0.25f * v_prev + 0.5f * v_curr + 0.25f * v_next;
                out->data[(row_curr + x) * ch + c] = (unsigned char)(sum < 0.0f ? 0 : (sum > 255.0f ? 255 : (unsigned char)(sum + 0.5f)));
            }
        }
    }
}

// Multithreaded OpenMP Separable Convolution (Fixed-point Integer Shift)
void conv_separable_omp_int(const Image *in, Image *out, uint16_t *temp) {
    int W = in->width;
    int H = in->height;
    int ch = in->channels;

    // Pass 1: Horizontal Blur parallelized over rows
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; y++) {
        int row_idx = y * W;
        for (int x = 0; x < W; x++) {
            int x_prev = (x > 0) ? x - 1 : 0;
            int x_next = (x < W - 1) ? x + 1 : W - 1;
            for (int c = 0; c < ch; c++) {
                uint16_t p0 = in->data[(row_idx + x_prev) * ch + c];
                uint16_t p1 = in->data[(row_idx + x) * ch + c];
                uint16_t p2 = in->data[(row_idx + x_next) * ch + c];
                temp[(row_idx + x) * ch + c] = (p0 + (p1 << 1) + p2 + 2) >> 2;
            }
        }
    }

    // Pass 2: Vertical Blur parallelized over rows
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; y++) {
        int y_prev = (y > 0) ? y - 1 : 0;
        int y_next = (y < H - 1) ? y + 1 : H - 1;
        int row_prev = y_prev * W;
        int row_curr = y * W;
        int row_next = y_next * W;

        for (int x = 0; x < W; x++) {
            for (int c = 0; c < ch; c++) {
                uint16_t p0 = temp[(row_prev + x) * ch + c];
                uint16_t p1 = temp[(row_curr + x) * ch + c];
                uint16_t p2 = temp[(row_next + x) * ch + c];
                out->data[(row_curr + x) * ch + c] = (unsigned char)((p0 + (p1 << 1) + p2 + 2) >> 2);
            }
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <mode: float|int> <input_image> <output_image>\n", argv[0]);
        return 1;
    }

    const char *mode = argv[1];
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
    int total_bytes = in.width * in.height * in.channels;
    out.data = (unsigned char *)malloc(total_bytes);

    double start_time = get_time_sec();
    if (strcmp(mode, "int") == 0) {
        uint16_t *temp = (uint16_t *)malloc(total_bytes * sizeof(uint16_t));
        conv_separable_omp_int(&in, &out, temp);
        free(temp);
    } else {
        float *temp = (float *)malloc(total_bytes * sizeof(float));
        conv_separable_omp_float(&in, &out, temp);
        free(temp);
    }
    double elapsed = get_time_sec() - start_time;

    if (strcmp(output_path, "none") != 0) {
        stbi_write_jpg(output_path, out.width, out.height, out.channels, out.data, 90);
    }

    free(in.data);
    free(out.data);

    printf("EXECUTION_TIME: %.6f\n", elapsed);
    return 0;
}
