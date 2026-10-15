#ifndef COMMON_H
#define COMMON_H

#if defined(__has_include) && __has_include("c_common.h")
#include "c_common.h"
#else
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <math.h>

typedef struct {
    int width;
    int height;
    int channels;
    unsigned char *data;
} Image;

typedef struct {
    int x_start;
    int x_end;
    int y_start;
    int y_end;
} Region;

static inline double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static inline int get_num_cores(void) {
    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
    return (nprocs > 0) ? (int)nprocs : 4;
}

static inline void get_grid_dimensions(int n, int *rows, int *cols) {
    int best_r = 1, best_c = n;
    for (int r = 1; r * r <= n; r++) {
        if (n % r == 0) {
            best_r = r;
            best_c = n / r;
        }
    }
    *rows = best_r;
    *cols = best_c;
}
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// 3x3 Gaussian Blur Kernel
static const float KERNEL[3][3] = {
    {1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f},
    {2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f},
    {1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f}
};

// Generate regions for Techniques 1-4
// Technique 1: Equal n regions (R x C grid based on cores)
// Technique 2: 9 regions (3x3 grid)
// Technique 3: n horizontal strip regions (Row-based)
// Technique 4: n vertical strip regions (Column-based)
static inline Region* get_regions(int technique, int width, int height, int *num_regions) {
    int n = get_num_cores();
    Region *regions = NULL;

    if (technique == 1) {
        int R, C;
        get_grid_dimensions(n, &R, &C);
        *num_regions = R * C;
        regions = (Region*)malloc(sizeof(Region) * (*num_regions));
        for (int r = 0; r < R; r++) {
            for (int c = 0; c < C; c++) {
                int idx = r * C + c;
                regions[idx].y_start = r * (height / R);
                regions[idx].y_end = (r == R - 1) ? height : (r + 1) * (height / R);
                regions[idx].x_start = c * (width / C);
                regions[idx].x_end = (c == C - 1) ? width : (c + 1) * (width / C);
            }
        }
    } else if (technique == 2) {
        *num_regions = 9;
        regions = (Region*)malloc(sizeof(Region) * 9);
        // Region 1: Top-left corner (1x1)
        regions[0].x_start = 0; regions[0].x_end = 1;
        regions[0].y_start = 0; regions[0].y_end = 1;

        // Region 2: Top edge (width-2 x 1)
        regions[1].x_start = 1; regions[1].x_end = width - 1;
        regions[1].y_start = 0; regions[1].y_end = 1;

        // Region 3: Top-right corner (1x1)
        regions[2].x_start = width - 1; regions[2].x_end = width;
        regions[2].y_start = 0; regions[2].y_end = 1;

        // Region 4: Left edge (1 x height-2)
        regions[3].x_start = 0; regions[3].x_end = 1;
        regions[3].y_start = 1; regions[3].y_end = height - 1;

        // Region 5: Center region (width-2 x height-2)
        regions[4].x_start = 1; regions[4].x_end = width - 1;
        regions[4].y_start = 1; regions[4].y_end = height - 1;

        // Region 6: Right edge (1 x height-2)
        regions[5].x_start = width - 1; regions[5].x_end = width;
        regions[5].y_start = 1; regions[5].y_end = height - 1;

        // Region 7: Bottom-left corner (1x1)
        regions[6].x_start = 0; regions[6].x_end = 1;
        regions[6].y_start = height - 1; regions[6].y_end = height;

        // Region 8: Bottom edge (width-2 x 1)
        regions[7].x_start = 1; regions[7].x_end = width - 1;
        regions[7].y_start = height - 1; regions[7].y_end = height;

        // Region 9: Bottom-right corner (1x1)
        regions[8].x_start = width - 1; regions[8].x_end = width;
        regions[8].y_start = height - 1; regions[8].y_end = height;
    } else if (technique == 3) {
        *num_regions = n;
        regions = (Region*)malloc(sizeof(Region) * n);
        for (int r = 0; r < n; r++) {
            regions[r].y_start = r * (height / n);
            regions[r].y_end = (r == n - 1) ? height : (r + 1) * (height / n);
            regions[r].x_start = 0;
            regions[r].x_end = width;
        }
    } else if (technique == 4) {
        *num_regions = n;
        regions = (Region*)malloc(sizeof(Region) * n);
        for (int c = 0; c < n; c++) {
            regions[c].y_start = 0;
            regions[c].y_end = height;
            regions[c].x_start = c * (width / n);
            regions[c].x_end = (c == n - 1) ? width : (c + 1) * (width / n);
        }
    }

    return regions;
}

static inline void apply_convolution_region(const Image *in, Image *out, Region reg) {
    int W = in->width;
    int H = in->height;
    int ch = in->channels;

    for (int y = reg.y_start; y < reg.y_end; y++) {
        for (int x = reg.x_start; x < reg.x_end; x++) {
            for (int c = 0; c < ch; c++) {
                float sum = 0.0f;
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        int py = y + ky;
                        int px = x + kx;
                        if (py < 0) py = 0; if (py >= H) py = H - 1;
                        if (px < 0) px = 0; if (px >= W) px = W - 1;
                        sum += in->data[(py * W + px) * ch + c] * KERNEL[ky + 1][kx + 1];
                    }
                }
                out->data[(y * W + x) * ch + c] = (unsigned char)(sum < 0.0f ? 0 : (sum > 255.0f ? 255 : (unsigned char)(sum + 0.5f)));
            }
        }
    }
}

#endif // COMMON_H
