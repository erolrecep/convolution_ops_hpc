#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arm_neon.h>
#include "../include/common.h"

// ARM NEON Vectorized 1D Horizontal Blur
// Processes 16 bytes (or 8 pixels RGB) per cycle using NEON registers
void neon_horizontal_blur_row(const uint8_t *in_row, uint8_t *out_row, int W, int ch) {
    int row_len = W * ch;
    // Process 16 bytes per iteration
    int x = 0;
    
    // Boundary pixel 0
    for (int c = 0; c < ch; c++) {
        uint16_t p0 = in_row[c];
        uint16_t p1 = in_row[c];
        uint16_t p2 = in_row[ch + c];
        out_row[c] = (p0 + (p1 << 1) + p2 + 2) >> 2;
    }

    // Main vector loop: process 16 bytes at a time
    for (x = ch; x <= row_len - ch - 16; x += 16) {
        uint8x16_t p0 = vld1q_u8(&in_row[x - ch]);
        uint8x16_t p1 = vld1q_u8(&in_row[x]);
        uint8x16_t p2 = vld1q_u8(&in_row[x + ch]);

        // Low 8 bytes
        uint16x8_t sum_low = vaddl_u8(vget_low_u8(p0), vget_low_u8(p2));
        sum_low = vaddq_u16(sum_low, vshll_n_u8(vget_low_u8(p1), 1));
        uint8x8_t res_low = vrshrn_n_u16(sum_low, 2);

        // High 8 bytes
        uint16x8_t sum_high = vaddl_u8(vget_high_u8(p0), vget_high_u8(p2));
        sum_high = vaddq_u16(sum_high, vshll_n_u8(vget_high_u8(p1), 1));
        uint8x8_t res_high = vrshrn_n_u16(sum_high, 2);

        // Combine and store 16 bytes
        uint8x16_t res = vcombine_u8(res_low, res_high);
        vst1q_u8(&out_row[x], res);
    }

    // Remaining scalar bytes
    for (; x < row_len; x++) {
        int c = x % ch;
        int col = x / ch;
        int col_prev = (col > 0) ? col - 1 : 0;
        int col_next = (col < W - 1) ? col + 1 : W - 1;
        uint16_t p0 = in_row[col_prev * ch + c];
        uint16_t p1 = in_row[col * ch + c];
        uint16_t p2 = in_row[col_next * ch + c];
        out_row[x] = (p0 + (p1 << 1) + p2 + 2) >> 2;
    }
}

// ARM NEON Vectorized 1D Vertical Blur
void neon_vertical_blur_row(const uint8_t *row_prev, const uint8_t *row_curr, const uint8_t *row_next, uint8_t *out_row, int row_len) {
    int x = 0;
    for (x = 0; x <= row_len - 16; x += 16) {
        uint8x16_t p0 = vld1q_u8(&row_prev[x]);
        uint8x16_t p1 = vld1q_u8(&row_curr[x]);
        uint8x16_t p2 = vld1q_u8(&row_next[x]);

        uint16x8_t sum_low = vaddl_u8(vget_low_u8(p0), vget_low_u8(p2));
        sum_low = vaddq_u16(sum_low, vshll_n_u8(vget_low_u8(p1), 1));
        uint8x8_t res_low = vrshrn_n_u16(sum_low, 2);

        uint16x8_t sum_high = vaddl_u8(vget_high_u8(p0), vget_high_u8(p2));
        sum_high = vaddq_u16(sum_high, vshll_n_u8(vget_high_u8(p1), 1));
        uint8x8_t res_high = vrshrn_n_u16(sum_high, 2);

        uint8x16_t res = vcombine_u8(res_low, res_high);
        vst1q_u8(&out_row[x], res);
    }
    for (; x < row_len; x++) {
        uint16_t p0 = row_prev[x];
        uint16_t p1 = row_curr[x];
        uint16_t p2 = row_next[x];
        out_row[x] = (p0 + (p1 << 1) + p2 + 2) >> 2;
    }
}

void conv_neon_separable(const Image *in, Image *out, uint8_t *temp) {
    int W = in->width;
    int H = in->height;
    int ch = in->channels;
    int row_len = W * ch;

    // Pass 1: Horizontal Blur (in -> temp)
    for (int y = 0; y < H; y++) {
        neon_horizontal_blur_row(&in->data[y * row_len], &temp[y * row_len], W, ch);
    }

    // Pass 2: Vertical Blur (temp -> out)
    for (int y = 0; y < H; y++) {
        int y_prev = (y > 0) ? y - 1 : 0;
        int y_next = (y < H - 1) ? y + 1 : H - 1;
        neon_vertical_blur_row(&temp[y_prev * row_len], &temp[y * row_len], &temp[y_next * row_len], &out->data[y * row_len], row_len);
    }
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
    Image out = in;
    out.data = (unsigned char *)malloc(total_bytes);
    uint8_t *temp = (uint8_t *)malloc(total_bytes);

    int ITERS = 5;
    double t_neon = 0.0;
    for (int i = 0; i < ITERS; i++) {
        double t0 = get_time_sec();
        conv_neon_separable(&in, &out, temp);
        t_neon += get_time_sec() - t0;
    }
    t_neon /= ITERS;

    double mp = (in.width * in.height / 1e6) / t_neon;
    printf("ARM NEON Vectorized Separable (Single Core): %.4fs (%.1f MP/s)\n", t_neon, mp);

    free(in.data);
    free(out.data);
    free(temp);
    return 0;
}
