#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../include/common.h"

// 1. Baseline 2D 3x3 convolution
void conv_2d_baseline(const Image *in, Image *out) {
    Region reg = {0, in->width, 0, in->height};
    apply_convolution_region(in, out, reg);
}

// 2. Separable Float: Pass 1 (Horizontal) + Pass 2 (Vertical)
void conv_separable_float(const Image *in, Image *out, float *temp) {
    int W = in->width;
    int H = in->height;
    int ch = in->channels;

    // Pass 1: Horizontal Blur (in -> temp)
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int x_prev = (x > 0) ? x - 1 : 0;
            int x_next = (x < W - 1) ? x + 1 : W - 1;
            int row_idx = y * W;
            for (int c = 0; c < ch; c++) {
                float v_prev = in->data[(row_idx + x_prev) * ch + c];
                float v_curr = in->data[(row_idx + x) * ch + c];
                float v_next = in->data[(row_idx + x_next) * ch + c];
                temp[(row_idx + x) * ch + c] = 0.25f * v_prev + 0.5f * v_curr + 0.25f * v_next;
            }
        }
    }

    // Pass 2: Vertical Blur (temp -> out)
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

// 3. Separable Integer Bit-Shift (Fixed Point, zero float math)
// 1/4 * (a + 2b + c) = (a + 2b + c + 2) >> 2
void conv_separable_int(const Image *in, Image *out, uint16_t *temp) {
    int W = in->width;
    int H = in->height;
    int ch = in->channels;

    // Pass 1: Horizontal Blur (in -> temp), result in [0, 255]
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

    // Pass 2: Vertical Blur (temp -> out)
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

// 4. Verification & Metric Check
int compare_images(const Image *a, const Image *b, double *max_diff, double *avg_diff) {
    int total = a->width * a->height * a->channels;
    double sum = 0.0;
    double max_d = 0.0;
    int mismatches = 0;

    for (int i = 0; i < total; i++) {
        double diff = fabs((double)a->data[i] - (double)b->data[i]);
        if (diff > max_d) max_d = diff;
        sum += diff;
        if (diff > 1.0) mismatches++; // Allow +/-1 LSB rounding differences
    }

    *max_diff = max_d;
    *avg_diff = sum / total;
    return mismatches;
}

int main(int argc, char **argv) {
    const char *img_path = (argc > 1) ? argv[1] : "images/bmw-m3-touring-24h-7680x4320-26531.jpg";

    Image in;
    in.data = stbi_load(img_path, &in.width, &in.height, &in.channels, 0);
    if (!in.data) {
        fprintf(stderr, "Could not open %s\n", img_path);
        return 1;
    }
    printf("Loaded image: %dx%d (%d channels)\n", in.width, in.height, in.channels);

    int total_bytes = in.width * in.height * in.channels;
    Image out_baseline = in, out_sep_float = in, out_sep_int = in;
    out_baseline.data = (unsigned char *)malloc(total_bytes);
    out_sep_float.data = (unsigned char *)malloc(total_bytes);
    out_sep_int.data = (unsigned char *)malloc(total_bytes);

    float *temp_float = (float *)malloc(total_bytes * sizeof(float));
    uint16_t *temp_int = (uint16_t *)malloc(total_bytes * sizeof(uint16_t));

    // Warm-up & timed runs (3 iterations)
    double t_base = 0.0, t_sep_float = 0.0, t_sep_int = 0.0;
    int ITERS = 3;

    printf("\nRunning %d benchmark iterations per algorithm...\n", ITERS);

    for (int i = 0; i < ITERS; i++) {
        double t0 = get_time_sec();
        conv_2d_baseline(&in, &out_baseline);
        t_base += get_time_sec() - t0;
    }
    t_base /= ITERS;
    printf("1. Baseline 2D 3x3:       %.4fs (%.1f MP/s)\n", t_base, (in.width * in.height / 1e6) / t_base);

    for (int i = 0; i < ITERS; i++) {
        double t0 = get_time_sec();
        conv_separable_float(&in, &out_sep_float, temp_float);
        t_sep_float += get_time_sec() - t0;
    }
    t_sep_float /= ITERS;
    printf("2. Separable Float (1D):   %.4fs (%.1f MP/s) -> Speedup: %.2fx\n", 
           t_sep_float, (in.width * in.height / 1e6) / t_sep_float, t_base / t_sep_float);

    for (int i = 0; i < ITERS; i++) {
        double t0 = get_time_sec();
        conv_separable_int(&in, &out_sep_int, temp_int);
        t_sep_int += get_time_sec() - t0;
    }
    t_sep_int /= ITERS;
    printf("3. Separable Integer/Shift: %.4fs (%.1f MP/s) -> Speedup: %.2fx\n", 
           t_sep_int, (in.width * in.height / 1e6) / t_sep_int, t_base / t_sep_int);

    // Accuracy Check
    double max_d = 0.0, avg_d = 0.0;
    int mismatches = compare_images(&out_baseline, &out_sep_float, &max_d, &avg_d);
    printf("\n--- Accuracy Verification (vs 2D Baseline) ---\n");
    printf("Separable Float: Max diff = %.1f, Avg diff = %.4f, Mismatches (>1 LSB) = %d\n", max_d, avg_d, mismatches);

    mismatches = compare_images(&out_baseline, &out_sep_int, &max_d, &avg_d);
    printf("Separable Int:   Max diff = %.1f, Avg diff = %.4f, Mismatches (>1 LSB) = %d\n", max_d, avg_d, mismatches);

    free(in.data);
    free(out_baseline.data);
    free(out_sep_float.data);
    free(out_sep_int.data);
    free(temp_float);
    free(temp_int);
    return 0;
}
