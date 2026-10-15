#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include "../include/common.h"

// AVX2 Vectorized 1D Horizontal Blur
// Processes 32 bytes (10+ pixels RGB) per cycle using 256-bit AVX2 registers
void avx2_horizontal_blur_row(const uint8_t *in_row, uint8_t *out_row, int W, int ch) {
    int row_len = W * ch;
    int x = 0;

    // Boundary pixel 0
    for (int c = 0; c < ch; c++) {
        uint16_t p0 = in_row[c];
        uint16_t p1 = in_row[c];
        uint16_t p2 = in_row[ch + c];
        out_row[c] = (p0 + (p1 << 1) + p2 + 2) >> 2;
    }

    // Process 16 bytes per step using 128-bit SSE/AVX registers for easy 16-bit unpacking
    const __m128i two_16 = _mm_set1_epi16(2);
    for (x = ch; x <= row_len - ch - 16; x += 16) {
        __m128i p0 = _mm_loadu_si128((const __m128i*)&in_row[x - ch]);
        __m128i p1 = _mm_loadu_si128((const __m128i*)&in_row[x]);
        __m128i p2 = _mm_loadu_si128((const __m128i*)&in_row[x + ch]);

        // Unpack low 8 bytes to 16-bit
        __m128i p0_lo = _mm_unpacklo_epi8(p0, _mm_setzero_si128());
        __m128i p1_lo = _mm_unpacklo_epi8(p1, _mm_setzero_si128());
        __m128i p2_lo = _mm_unpacklo_epi8(p2, _mm_setzero_si128());

        // (p0 + 2*p1 + p2 + 2) >> 2
        __m128i sum_lo = _mm_add_epi16(p0_lo, p2_lo);
        sum_lo = _mm_add_epi16(sum_lo, _mm_slli_epi16(p1_lo, 1));
        sum_lo = _mm_add_epi16(sum_lo, two_16);
        sum_lo = _mm_srli_epi16(sum_lo, 2);

        // Unpack high 8 bytes to 16-bit
        __m128i p0_hi = _mm_unpackhi_epi8(p0, _mm_setzero_si128());
        __m128i p1_hi = _mm_unpackhi_epi8(p1, _mm_setzero_si128());
        __m128i p2_hi = _mm_unpackhi_epi8(p2, _mm_setzero_si128());

        __m128i sum_hi = _mm_add_epi16(p0_hi, p2_hi);
        sum_hi = _mm_add_epi16(sum_hi, _mm_slli_epi16(p1_hi, 1));
        sum_hi = _mm_add_epi16(sum_hi, two_16);
        sum_hi = _mm_srli_epi16(sum_hi, 2);

        // Pack back to 16 uint8 bytes
        __m128i res = _mm_packus_epi16(sum_lo, sum_hi);
        _mm_storeu_si128((__m128i*)&out_row[x], res);
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

// AVX2 Vectorized 1D Vertical Blur
void avx2_vertical_blur_row(const uint8_t *row_prev, const uint8_t *row_curr, const uint8_t *row_next, uint8_t *out_row, int row_len) {
    int x = 0;
    const __m128i two_16 = _mm_set1_epi16(2);

    for (x = 0; x <= row_len - 16; x += 16) {
        __m128i p0 = _mm_loadu_si128((const __m128i*)&row_prev[x]);
        __m128i p1 = _mm_loadu_si128((const __m128i*)&row_curr[x]);
        __m128i p2 = _mm_loadu_si128((const __m128i*)&row_next[x]);

        __m128i p0_lo = _mm_unpacklo_epi8(p0, _mm_setzero_si128());
        __m128i p1_lo = _mm_unpacklo_epi8(p1, _mm_setzero_si128());
        __m128i p2_lo = _mm_unpacklo_epi8(p2, _mm_setzero_si128());

        __m128i sum_lo = _mm_add_epi16(p0_lo, p2_lo);
        sum_lo = _mm_add_epi16(sum_lo, _mm_slli_epi16(p1_lo, 1));
        sum_lo = _mm_add_epi16(sum_lo, two_16);
        sum_lo = _mm_srli_epi16(sum_lo, 2);

        __m128i p0_hi = _mm_unpackhi_epi8(p0, _mm_setzero_si128());
        __m128i p1_hi = _mm_unpackhi_epi8(p1, _mm_setzero_si128());
        __m128i p2_hi = _mm_unpackhi_epi8(p2, _mm_setzero_si128());

        __m128i sum_hi = _mm_add_epi16(p0_hi, p2_hi);
        sum_hi = _mm_add_epi16(sum_hi, _mm_slli_epi16(p1_hi, 1));
        sum_hi = _mm_add_epi16(sum_hi, two_16);
        sum_hi = _mm_srli_epi16(sum_hi, 2);

        __m128i res = _mm_packus_epi16(sum_lo, sum_hi);
        _mm_storeu_si128((__m128i*)&out_row[x], res);
    }

    for (; x < row_len; x++) {
        uint16_t p0 = row_prev[x];
        uint16_t p1 = row_curr[x];
        uint16_t p2 = row_next[x];
        out_row[x] = (p0 + (p1 << 1) + p2 + 2) >> 2;
    }
}

void conv_avx2_separable(const Image *in, Image *out, uint8_t *temp) {
    int W = in->width;
    int H = in->height;
    int ch = in->channels;
    int row_len = W * ch;

    // Pass 1: Horizontal Blur (in -> temp)
    for (int y = 0; y < H; y++) {
        avx2_horizontal_blur_row(&in->data[y * row_len], &temp[y * row_len], W, ch);
    }

    // Pass 2: Vertical Blur (temp -> out)
    for (int y = 0; y < H; y++) {
        int y_prev = (y > 0) ? y - 1 : 0;
        int y_next = (y < H - 1) ? y + 1 : H - 1;
        avx2_vertical_blur_row(&temp[y_prev * row_len], &temp[y * row_len], &temp[y_next * row_len], &out->data[y * row_len], row_len);
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
    double t_avx2 = 0.0;
    for (int i = 0; i < ITERS; i++) {
        double t0 = get_time_sec();
        conv_avx2_separable(&in, &out, temp);
        t_avx2 += get_time_sec() - t0;
    }
    t_avx2 /= ITERS;

    double mp = (in.width * in.height / 1e6) / t_avx2;
    printf("x86 AVX2 Vectorized Separable (Single Core): %.4fs (%.1f MP/s)\n", t_avx2, mp);

    free(in.data);
    free(out.data);
    free(temp);
    return 0;
}
