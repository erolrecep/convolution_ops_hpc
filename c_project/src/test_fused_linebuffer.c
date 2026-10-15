#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif
#include <omp.h>
#include "../include/common.h"

// Horizontal blur for a single row using NEON or scalar fallback
static inline void horizontal_blur_row(const uint8_t *in_row, uint8_t *out_row, int W, int ch) {
    int row_len = W * ch;
    int x = 0;

#if defined(__ARM_NEON) || defined(__aarch64__)
    // First pixel boundary
    for (int c = 0; c < ch; c++) {
        uint16_t p0 = in_row[c];
        uint16_t p1 = in_row[c];
        uint16_t p2 = in_row[ch + c];
        out_row[c] = (p0 + (p1 << 1) + p2 + 2) >> 2;
    }

    // Vector NEON loop (16 bytes per cycle)
    for (x = ch; x <= row_len - ch - 16; x += 16) {
        uint8x16_t p0 = vld1q_u8(&in_row[x - ch]);
        uint8x16_t p1 = vld1q_u8(&in_row[x]);
        uint8x16_t p2 = vld1q_u8(&in_row[x + ch]);

        uint16x8_t sum_low = vaddl_u8(vget_low_u8(p0), vget_low_u8(p2));
        sum_low = vaddq_u16(sum_low, vshll_n_u8(vget_low_u8(p1), 1));
        uint8x8_t res_low = vrshrn_n_u16(sum_low, 2);

        uint16x8_t sum_high = vaddl_u8(vget_high_u8(p0), vget_high_u8(p2));
        sum_high = vaddq_u16(sum_high, vshll_n_u8(vget_high_u8(p1), 1));
        uint8x8_t res_high = vrshrn_n_u16(sum_high, 2);

        uint8x16_t res = vcombine_u8(res_low, res_high);
        vst1q_u8(&out_row[x], res);
    }
#elif defined(__AVX2__)
    // First pixel boundary
    for (int c = 0; c < ch; c++) {
        uint16_t p0 = in_row[c];
        uint16_t p1 = in_row[c];
        uint16_t p2 = in_row[ch + c];
        out_row[c] = (p0 + (p1 << 1) + p2 + 2) >> 2;
    }
    const __m128i two_16 = _mm_set1_epi16(2);
    for (x = ch; x <= row_len - ch - 16; x += 16) {
        __m128i p0 = _mm_loadu_si128((const __m128i*)&in_row[x - ch]);
        __m128i p1 = _mm_loadu_si128((const __m128i*)&in_row[x]);
        __m128i p2 = _mm_loadu_si128((const __m128i*)&in_row[x + ch]);

        __m128i p0_lo = _mm_unpacklo_epi8(p0, _mm_setzero_si128());
        __m128i p1_lo = _mm_unpacklo_epi8(p1, _mm_setzero_si128());
        __m128i p2_lo = _mm_unpacklo_epi8(p2, _mm_setzero_si128());
        __m128i sum_lo = _mm_srli_epi16(_mm_add_epi16(_mm_add_epi16(_mm_add_epi16(p0_lo, p2_lo), _mm_slli_epi16(p1_lo, 1)), two_16), 2);

        __m128i p0_hi = _mm_unpackhi_epi8(p0, _mm_setzero_si128());
        __m128i p1_hi = _mm_unpackhi_epi8(p1, _mm_setzero_si128());
        __m128i p2_hi = _mm_unpackhi_epi8(p2, _mm_setzero_si128());
        __m128i sum_hi = _mm_srli_epi16(_mm_add_epi16(_mm_add_epi16(_mm_add_epi16(p0_hi, p2_hi), _mm_slli_epi16(p1_hi, 1)), two_16), 2);

        __m128i res = _mm_packus_epi16(sum_lo, sum_hi);
        _mm_storeu_si128((__m128i*)&out_row[x], res);
    }
#endif

    // Scalar tail
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

// Vertical blur between 3 consecutive rows
static inline void vertical_blur_row(const uint8_t *r_prev, const uint8_t *r_curr, const uint8_t *r_next, uint8_t *out_row, int row_len) {
    int x = 0;

#if defined(__ARM_NEON) || defined(__aarch64__)
    for (x = 0; x <= row_len - 16; x += 16) {
        uint8x16_t p0 = vld1q_u8(&r_prev[x]);
        uint8x16_t p1 = vld1q_u8(&r_curr[x]);
        uint8x16_t p2 = vld1q_u8(&r_next[x]);

        uint16x8_t sum_low = vaddl_u8(vget_low_u8(p0), vget_low_u8(p2));
        sum_low = vaddq_u16(sum_low, vshll_n_u8(vget_low_u8(p1), 1));
        uint8x8_t res_low = vrshrn_n_u16(sum_low, 2);

        uint16x8_t sum_high = vaddl_u8(vget_high_u8(p0), vget_high_u8(p2));
        sum_high = vaddq_u16(sum_high, vshll_n_u8(vget_high_u8(p1), 1));
        uint8x8_t res_high = vrshrn_n_u16(sum_high, 2);

        uint8x16_t res = vcombine_u8(res_low, res_high);
        vst1q_u8(&out_row[x], res);
    }
#elif defined(__AVX2__)
    const __m128i two_16 = _mm_set1_epi16(2);
    for (x = 0; x <= row_len - 16; x += 16) {
        __m128i p0 = _mm_loadu_si128((const __m128i*)&r_prev[x]);
        __m128i p1 = _mm_loadu_si128((const __m128i*)&r_curr[x]);
        __m128i p2 = _mm_loadu_si128((const __m128i*)&r_next[x]);

        __m128i p0_lo = _mm_unpacklo_epi8(p0, _mm_setzero_si128());
        __m128i p1_lo = _mm_unpacklo_epi8(p1, _mm_setzero_si128());
        __m128i p2_lo = _mm_unpacklo_epi8(p2, _mm_setzero_si128());
        __m128i sum_lo = _mm_srli_epi16(_mm_add_epi16(_mm_add_epi16(_mm_add_epi16(p0_lo, p2_lo), _mm_slli_epi16(p1_lo, 1)), two_16), 2);

        __m128i p0_hi = _mm_unpackhi_epi8(p0, _mm_setzero_si128());
        __m128i p1_hi = _mm_unpackhi_epi8(p1, _mm_setzero_si128());
        __m128i p2_hi = _mm_unpackhi_epi8(p2, _mm_setzero_si128());
        __m128i sum_hi = _mm_srli_epi16(_mm_add_epi16(_mm_add_epi16(_mm_add_epi16(p0_hi, p2_hi), _mm_slli_epi16(p1_hi, 1)), two_16), 2);

        __m128i res = _mm_packus_epi16(sum_lo, sum_hi);
        _mm_storeu_si128((__m128i*)&out_row[x], res);
    }
#endif

    for (; x < row_len; x++) {
        uint16_t p0 = r_prev[x];
        uint16_t p1 = r_curr[x];
        uint16_t p2 = r_next[x];
        out_row[x] = (p0 + (p1 << 1) + p2 + 2) >> 2;
    }
}

// Level 4 Fused Ring-Buffer Separable Convolution (Single-Threaded)
// Intermediate buffer is only 3 lines (~69 KB), 100% resident in L1/L2 cache!
void conv_fused_linebuffer_serial(const Image *in, Image *out) {
    int W = in->width;
    int H = in->height;
    int ch = in->channels;
    int row_len = W * ch;

    // Cache-resident 3-line rolling ring buffer (69 KB)
    uint8_t *ring[3];
    ring[0] = (uint8_t *)malloc(row_len);
    ring[1] = (uint8_t *)malloc(row_len);
    ring[2] = (uint8_t *)malloc(row_len);

    // Bootstrap ring buffer with rows 0 and 1
    horizontal_blur_row(&in->data[0], ring[0], W, ch);
    horizontal_blur_row(&in->data[row_len], ring[1], W, ch);

    // Top border row 0: r_prev is row 0
    vertical_blur_row(ring[0], ring[0], ring[1], &out->data[0], row_len);

    // Process middle rows 1 to H-2
    for (int y = 1; y < H - 1; y++) {
        // Read next row into ring buffer slot
        int next_slot = (y + 1) % 3;
        int curr_slot = y % 3;
        int prev_slot = (y - 1) % 3;

        horizontal_blur_row(&in->data[(y + 1) * row_len], ring[next_slot], W, ch);
        vertical_blur_row(ring[prev_slot], ring[curr_slot], ring[next_slot], &out->data[y * row_len], row_len);
    }

    // Bottom border row H-1: r_next is row H-1
    int last_curr = (H - 1) % 3;
    int last_prev = (H - 2) % 3;
    vertical_blur_row(ring[last_prev], ring[last_curr], ring[last_curr], &out->data[(H - 1) * row_len], row_len);

    free(ring[0]);
    free(ring[1]);
    free(ring[2]);
}

// Level 4 Fused Ring-Buffer Separable Convolution (OpenMP Multi-Core)
// Each thread gets its own private 3-line L1 cache buffer for its assigned band!
void conv_fused_linebuffer_omp(const Image *in, Image *out) {
    int W = in->width;
    int H = in->height;
    int ch = in->channels;
    int row_len = W * ch;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        int y_start = tid * (H / nthreads);
        int y_end = (tid == nthreads - 1) ? H : (tid + 1) * (H / nthreads);

        uint8_t *ring[3];
        ring[0] = (uint8_t *)malloc(row_len);
        ring[1] = (uint8_t *)malloc(row_len);
        ring[2] = (uint8_t *)malloc(row_len);

        int y_prev_clamped = (y_start > 0) ? y_start - 1 : 0;
        int y_next_clamped = (y_start + 1 < H) ? y_start + 1 : H - 1;

        horizontal_blur_row(&in->data[y_prev_clamped * row_len], ring[0], W, ch);
        horizontal_blur_row(&in->data[y_start * row_len], ring[1], W, ch);
        horizontal_blur_row(&in->data[y_next_clamped * row_len], ring[2], W, ch);

        vertical_blur_row(ring[0], ring[1], ring[2], &out->data[y_start * row_len], row_len);

        for (int y = y_start + 1; y < y_end; y++) {
            int prev_slot = (y - 1) % 3;
            int curr_slot = y % 3;
            int next_slot = (y + 1) % 3;

            int next_y = (y + 1 < H) ? y + 1 : H - 1;
            horizontal_blur_row(&in->data[next_y * row_len], ring[next_slot], W, ch);
            vertical_blur_row(ring[prev_slot], ring[curr_slot], ring[next_slot], &out->data[y * row_len], row_len);
        }

        free(ring[0]);
        free(ring[1]);
        free(ring[2]);
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
    Image out_serial = in, out_omp = in;
    out_serial.data = (unsigned char *)malloc(total_bytes);
    out_omp.data = (unsigned char *)malloc(total_bytes);

    int ITERS = 10;
    double t_serial = 0.0, t_omp = 0.0;

    for (int i = 0; i < ITERS; i++) {
        double t0 = get_time_sec();
        conv_fused_linebuffer_serial(&in, &out_serial);
        t_serial += get_time_sec() - t0;
    }
    t_serial /= ITERS;

    for (int i = 0; i < ITERS; i++) {
        double t0 = get_time_sec();
        conv_fused_linebuffer_omp(&in, &out_omp);
        t_omp += get_time_sec() - t0;
    }
    t_omp /= ITERS;

    double mp_serial = (in.width * in.height / 1e6) / t_serial;
    double mp_omp = (in.width * in.height / 1e6) / t_omp;

    printf("\n--- Level 4: Cache-Resident Ring-Buffer Performance ---\n");
    printf("Fused Ring-Buffer (Single Core): %.4fs (%.1f MP/s)\n", t_serial, mp_serial);
    printf("Fused Ring-Buffer (Multi-Core):  %.4fs (%.1f MP/s)\n", t_omp, mp_omp);

    free(in.data);
    free(out_serial.data);
    free(out_omp.data);
    return 0;
}
