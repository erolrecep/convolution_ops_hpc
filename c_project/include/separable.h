#ifndef SEPARABLE_H
#define SEPARABLE_H

#include "common.h"
#include <stdint.h>

/**
 * High-Performance Separable 1D+1D Convolution (Gaussian 3x3)
 * Decomposes 2D 3x3 kernel into horizontal [0.25, 0.5, 0.25] and vertical [0.25, 0.5, 0.25] passes.
 * Reduces operations from 9 to 6 per pixel (33% arithmetic reduction).
 */

// Floating-point separable convolution (100% bit-exact with 2D Gaussian)
static inline void apply_separable_conv_float(const Image *in, Image *out, float *temp) {
    int W = in->width;
    int H = in->height;
    int ch = in->channels;

    // Pass 1: Horizontal Blur (in -> temp)
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

// Fixed-point Integer bit-shift separable convolution (maximum throughput)
static inline void apply_separable_conv_int(const Image *in, Image *out, uint16_t *temp) {
    int W = in->width;
    int H = in->height;
    int ch = in->channels;

    // Pass 1: Horizontal Blur (in -> temp)
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

#endif // SEPARABLE_H
