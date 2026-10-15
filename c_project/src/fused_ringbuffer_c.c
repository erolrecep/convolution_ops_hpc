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

// Horizontal blur for a single row slice [x_start, x_end) of image 'in'
static inline void horizontal_blur_row_slice(
    const uint8_t *in_full_row, uint8_t *out_slice,
    int x_start, int x_end, int full_width, int ch)
{
    int slice_w = x_end - x_start;
    int slice_bytes = slice_w * ch;
    int x = 0;

#if defined(__ARM_NEON) || defined(__aarch64__)
    // Handle first boundary pixel of slice
    for (int c = 0; c < ch; c++) {
        int x0 = (x_start > 0) ? (x_start - 1) : 0;
        int x1 = x_start;
        int x2 = (x_start + 1 < full_width) ? (x_start + 1) : (full_width - 1);
        uint16_t p0 = in_full_row[x0 * ch + c];
        uint16_t p1 = in_full_row[x1 * ch + c];
        uint16_t p2 = in_full_row[x2 * ch + c];
        out_slice[c] = (p0 + (p1 << 1) + p2 + 2) >> 2;
    }

    // Interior 16-byte vectorized NEON chunks
    for (x = ch; x <= slice_bytes - ch - 16; x += 16) {
        int gx = x_start * ch + x;
        uint8x16_t p0 = vld1q_u8(&in_full_row[gx - ch]);
        uint8x16_t p1 = vld1q_u8(&in_full_row[gx]);
        uint8x16_t p2 = vld1q_u8(&in_full_row[gx + ch]);

        uint16x8_t sum_low = vaddl_u8(vget_low_u8(p0), vget_low_u8(p2));
        sum_low = vaddq_u16(sum_low, vshll_n_u8(vget_low_u8(p1), 1));
        uint8x8_t res_low = vrshrn_n_u16(sum_low, 2);

        uint16x8_t sum_high = vaddl_u8(vget_high_u8(p0), vget_high_u8(p2));
        sum_high = vaddq_u16(sum_high, vshll_n_u8(vget_high_u8(p1), 1));
        uint8x8_t res_high = vrshrn_n_u16(sum_high, 2);

        vst1q_u8(&out_slice[x], vcombine_u8(res_low, res_high));
    }

    // Tail scalar loop
    for (; x < slice_bytes; x++) {
        int px = x_start + x / ch;
        int c = x % ch;
        int x0 = (px > 0) ? (px - 1) : 0;
        int x1 = px;
        int x2 = (px + 1 < full_width) ? (px + 1) : (full_width - 1);
        uint16_t p0 = in_full_row[x0 * ch + c];
        uint16_t p1 = in_full_row[x1 * ch + c];
        uint16_t p2 = in_full_row[x2 * ch + c];
        out_slice[x] = (p0 + (p1 << 1) + p2 + 2) >> 2;
    }
#elif defined(__AVX2__)
    for (int c = 0; c < ch; c++) {
        int x0 = (x_start > 0) ? (x_start - 1) : 0;
        int x1 = x_start;
        int x2 = (x_start + 1 < full_width) ? (x_start + 1) : (full_width - 1);
        uint16_t p0 = in_full_row[x0 * ch + c];
        uint16_t p1 = in_full_row[x1 * ch + c];
        uint16_t p2 = in_full_row[x2 * ch + c];
        out_slice[c] = (p0 + (p1 << 1) + p2 + 2) >> 2;
    }
    const __m128i two_16 = _mm_set1_epi16(2);
    for (x = ch; x <= slice_bytes - ch - 16; x += 16) {
        int gx = x_start * ch + x;
        __m128i p0 = _mm_loadu_si128((const __m128i*)&in_full_row[gx - ch]);
        __m128i p1 = _mm_loadu_si128((const __m128i*)&in_full_row[gx]);
        __m128i p2 = _mm_loadu_si128((const __m128i*)&in_full_row[gx + ch]);

        __m128i p0_lo = _mm_unpacklo_epi8(p0, _mm_setzero_si128());
        __m128i p1_lo = _mm_unpacklo_epi8(p1, _mm_setzero_si128());
        __m128i p2_lo = _mm_unpacklo_epi8(p2, _mm_setzero_si128());

        __m128i s_lo = _mm_add_epi16(_mm_add_epi16(p0_lo, _mm_slli_epi16(p1_lo, 1)), _mm_add_epi16(p2_lo, two_16));
        s_lo = _mm_srli_epi16(s_lo, 2);

        __m128i p0_hi = _mm_unpackhi_epi8(p0, _mm_setzero_si128());
        __m128i p1_hi = _mm_unpackhi_epi8(p1, _mm_setzero_si128());
        __m128i p2_hi = _mm_unpackhi_epi8(p2, _mm_setzero_si128());

        __m128i s_hi = _mm_add_epi16(_mm_add_epi16(p0_hi, _mm_slli_epi16(p1_hi, 1)), _mm_add_epi16(p2_hi, two_16));
        s_hi = _mm_srli_epi16(s_hi, 2);

        _mm_storeu_si128((__m128i*)&out_slice[x], _mm_packus_epi16(s_lo, s_hi));
    }
    for (; x < slice_bytes; x++) {
        int px = x_start + x / ch;
        int c = x % ch;
        int x0 = (px > 0) ? (px - 1) : 0;
        int x1 = px;
        int x2 = (px + 1 < full_width) ? (px + 1) : (full_width - 1);
        uint16_t p0 = in_full_row[x0 * ch + c];
        uint16_t p1 = in_full_row[x1 * ch + c];
        uint16_t p2 = in_full_row[x2 * ch + c];
        out_slice[x] = (p0 + (p1 << 1) + p2 + 2) >> 2;
    }
#else
    for (x = 0; x < slice_bytes; x++) {
        int px = x_start + x / ch;
        int c = x % ch;
        int x0 = (px > 0) ? (px - 1) : 0;
        int x1 = px;
        int x2 = (px + 1 < full_width) ? (px + 1) : (full_width - 1);
        uint16_t p0 = in_full_row[x0 * ch + c];
        uint16_t p1 = in_full_row[x1 * ch + c];
        uint16_t p2 = in_full_row[x2 * ch + c];
        out_slice[x] = (p0 + (p1 << 1) + p2 + 2) >> 2;
    }
#endif
}

// Vertical blur combining 3 horizontal slices into output slice
static inline void vertical_blur_slice(
    const uint8_t *r0, const uint8_t *r1, const uint8_t *r2,
    uint8_t *out_slice, int slice_bytes)
{
    int x = 0;

#if defined(__ARM_NEON) || defined(__aarch64__)
    for (x = 0; x <= slice_bytes - 16; x += 16) {
        uint8x16_t p0 = vld1q_u8(&r0[x]);
        uint8x16_t p1 = vld1q_u8(&r1[x]);
        uint8x16_t p2 = vld1q_u8(&r2[x]);

        uint16x8_t sum_low = vaddl_u8(vget_low_u8(p0), vget_low_u8(p2));
        sum_low = vaddq_u16(sum_low, vshll_n_u8(vget_low_u8(p1), 1));
        uint8x8_t res_low = vrshrn_n_u16(sum_low, 2);

        uint16x8_t sum_high = vaddl_u8(vget_high_u8(p0), vget_high_u8(p2));
        sum_high = vaddq_u16(sum_high, vshll_n_u8(vget_high_u8(p1), 1));
        uint8x8_t res_high = vrshrn_n_u16(sum_high, 2);

        vst1q_u8(&out_slice[x], vcombine_u8(res_low, res_high));
    }
#elif defined(__AVX2__)
    const __m128i two_16 = _mm_set1_epi16(2);
    for (x = 0; x <= slice_bytes - 16; x += 16) {
        __m128i p0 = _mm_loadu_si128((const __m128i*)&r0[x]);
        __m128i p1 = _mm_loadu_si128((const __m128i*)&r1[x]);
        __m128i p2 = _mm_loadu_si128((const __m128i*)&r2[x]);

        __m128i p0_lo = _mm_unpacklo_epi8(p0, _mm_setzero_si128());
        __m128i p1_lo = _mm_unpacklo_epi8(p1, _mm_setzero_si128());
        __m128i p2_lo = _mm_unpacklo_epi8(p2, _mm_setzero_si128());

        __m128i s_lo = _mm_add_epi16(_mm_add_epi16(p0_lo, _mm_slli_epi16(p1_lo, 1)), _mm_add_epi16(p2_lo, two_16));
        s_lo = _mm_srli_epi16(s_lo, 2);

        __m128i p0_hi = _mm_unpackhi_epi8(p0, _mm_setzero_si128());
        __m128i p1_hi = _mm_unpackhi_epi8(p1, _mm_setzero_si128());
        __m128i p2_hi = _mm_unpackhi_epi8(p2, _mm_setzero_si128());

        __m128i s_hi = _mm_add_epi16(_mm_add_epi16(p0_hi, _mm_slli_epi16(p1_hi, 1)), _mm_add_epi16(p2_hi, two_16));
        s_hi = _mm_srli_epi16(s_hi, 2);

        _mm_storeu_si128((__m128i*)&out_slice[x], _mm_packus_epi16(s_lo, s_hi));
    }
#endif

    for (; x < slice_bytes; x++) {
        uint16_t p0 = r0[x];
        uint16_t p1 = r1[x];
        uint16_t p2 = r2[x];
        out_slice[x] = (p0 + (p1 << 1) + p2 + 2) >> 2;
    }
}

// Process a rectangular region [x_start, x_end) x [y_start, y_end) using 3-line rolling ring-buffer
void process_region_ringbuffer(const Image *in, Image *out, Region reg) {
    int W = in->width;
    int H = in->height;
    int ch = in->channels;
    int full_row_len = W * ch;

    int slice_w = reg.x_end - reg.x_start;
    if (slice_w <= 0 || reg.y_end <= reg.y_start) return;
    int slice_bytes = slice_w * ch;

    uint8_t *ring[3];
    ring[0] = (uint8_t *)malloc(slice_bytes);
    ring[1] = (uint8_t *)malloc(slice_bytes);
    ring[2] = (uint8_t *)malloc(slice_bytes);

    int y_start = reg.y_start;
    int y_end = reg.y_end;

    int y_prev_clamped = (y_start > 0) ? y_start - 1 : 0;
    int y_next_clamped = (y_start + 1 < H) ? y_start + 1 : H - 1;

    horizontal_blur_row_slice(&in->data[y_prev_clamped * full_row_len], ring[0], reg.x_start, reg.x_end, W, ch);
    horizontal_blur_row_slice(&in->data[y_start * full_row_len], ring[1], reg.x_start, reg.x_end, W, ch);
    horizontal_blur_row_slice(&in->data[y_next_clamped * full_row_len], ring[2], reg.x_start, reg.x_end, W, ch);

    vertical_blur_slice(ring[0], ring[1], ring[2], &out->data[y_start * full_row_len + reg.x_start * ch], slice_bytes);

    for (int y = y_start + 1; y < y_end; y++) {
        int prev_slot = (y - 1) % 3;
        int curr_slot = y % 3;
        int next_slot = (y + 1) % 3;

        int next_y = (y + 1 < H) ? y + 1 : H - 1;
        horizontal_blur_row_slice(&in->data[next_y * full_row_len], ring[next_slot], reg.x_start, reg.x_end, W, ch);
        vertical_blur_slice(ring[prev_slot], ring[curr_slot], ring[next_slot],
                            &out->data[y * full_row_len + reg.x_start * ch], slice_bytes);
    }

    free(ring[0]);
    free(ring[1]);
    free(ring[2]);
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
    size_t total_bytes = (size_t)in.width * in.height * in.channels;
    out.data = (unsigned char *)malloc(total_bytes);

    int num_regions = 0;
    Region *regions = get_regions(technique, in.width, in.height, &num_regions);

    double start_time = get_time_sec();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_regions; i++) {
        process_region_ringbuffer(&in, &out, regions[i]);
    }

    double elapsed = get_time_sec() - start_time;

    if (strcmp(output_path, "none") != 0) {
        stbi_write_jpg(output_path, out.width, out.height, out.channels, out.data, 90);
    }

    free(regions);
    stbi_image_free(in.data);
    free(out.data);

    printf("EXECUTION_TIME: %.6f\n", elapsed);
    return 0;
}
