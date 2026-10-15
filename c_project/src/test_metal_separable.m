#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/common.h"

typedef struct {
    uint32_t x_start;
    uint32_t x_end;
    uint32_t y_start;
    uint32_t y_end;
    uint32_t img_width;
    uint32_t img_height;
    uint32_t channels;
} RegionParams;

static const char *metal_shader_src =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"struct RegionParams {\n"
"    uint x_start;\n"
"    uint x_end;\n"
"    uint y_start;\n"
"    uint y_end;\n"
"    uint img_width;\n"
"    uint img_height;\n"
"    uint channels;\n"
"};\n"
"\n"
"// Fused Threadgroup-Tiled Separable Convolution\n"
"// Threadgroup dimensions: 16 x 16 (256 threads)\n"
"// Shared tile dimensions: (16 + 2) x (16 + 2) = 18 x 18 pixels (each 3 channels)\n"
"kernel void conv_tiled_separable(\n"
"    device const uchar* in [[buffer(0)]],\n"
"    device uchar* out [[buffer(1)]],\n"
"    constant RegionParams& reg [[buffer(2)]],\n"
"    threadgroup uchar* s_tile [[threadgroup(0)]],\n"
"    uint2 tid [[thread_position_in_threadgroup]],\n"
"    uint2 gid [[threadgroup_position_in_grid]],\n"
"    uint2 t_pos [[thread_position_in_grid]])\n"
"{\n"
"    int W = reg.img_width;\n"
"    int H = reg.img_height;\n"
"    int ch = reg.channels;\n"
"\n"
"    int base_x = gid.x * 16 - 1;\n"
"    int base_y = gid.y * 16 - 1;\n"
"    int linear_tid = tid.y * 16 + tid.x;\n"
"\n"
"    // Load 18x18 = 324 pixels into threadgroup shared memory (324 * 3 = 972 bytes)\n"
"    // 256 threads load first 256 elements\n"
"    if (linear_tid < 256) {\n"
"        int sy = linear_tid / 18;\n"
"        int sx = linear_tid % 18;\n"
"        int gy = clamp(base_y + sy, 0, H - 1);\n"
"        int gx = clamp(base_x + sx, 0, W - 1);\n"
"        int s_idx = (sy * 18 + sx) * 3;\n"
"        int g_idx = (gy * W + gx) * 3;\n"
"        s_tile[s_idx + 0] = in[g_idx + 0];\n"
"        s_tile[s_idx + 1] = in[g_idx + 1];\n"
"        s_tile[s_idx + 2] = in[g_idx + 2];\n"
"    }\n"
"    // Load remaining 324 - 256 = 68 elements\n"
"    if (linear_tid < 68) {\n"
"        int idx = 256 + linear_tid;\n"
"        int sy = idx / 18;\n"
"        int sx = idx % 18;\n"
"        int gy = clamp(base_y + sy, 0, H - 1);\n"
"        int gx = clamp(base_x + sx, 0, W - 1);\n"
"        int s_idx = (sy * 18 + sx) * 3;\n"
"        int g_idx = (gy * W + gx) * 3;\n"
"        s_tile[s_idx + 0] = in[g_idx + 0];\n"
"        s_tile[s_idx + 1] = in[g_idx + 1];\n"
"        s_tile[s_idx + 2] = in[g_idx + 2];\n"
"    }\n"
"\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"\n"
"    int out_x = gid.x * 16 + tid.x;\n"
"    int out_y = gid.y * 16 + tid.y;\n"
"    if (out_x >= (int)reg.img_width || out_y >= (int)reg.img_height) return;\n"
"\n"
"    // Compute separable filter directly from threadgroup memory\n"
"    int sy = tid.y + 1;\n"
"    int sx = tid.x + 1;\n"
"\n"
"    for (int c = 0; c < 3; c++) {\n"
"        // Horizontal pass for 3 adjacent rows in shared memory\n"
"        ushort r0 = (ushort)s_tile[((sy - 1) * 18 + (sx - 1)) * 3 + c] +\n"
"                    ((ushort)s_tile[((sy - 1) * 18 + sx) * 3 + c] << 1) +\n"
"                    (ushort)s_tile[((sy - 1) * 18 + (sx + 1)) * 3 + c] + 2;\n"
"        r0 >>= 2;\n"
"\n"
"        ushort r1 = (ushort)s_tile[(sy * 18 + (sx - 1)) * 3 + c] +\n"
"                    ((ushort)s_tile[(sy * 18 + sx) * 3 + c] << 1) +\n"
"                    (ushort)s_tile[(sy * 18 + (sx + 1)) * 3 + c] + 2;\n"
"        r1 >>= 2;\n"
"\n"
"        ushort r2 = (ushort)s_tile[((sy + 1) * 18 + (sx - 1)) * 3 + c] +\n"
"                    ((ushort)s_tile[((sy + 1) * 18 + sx) * 3 + c] << 1) +\n"
"                    (ushort)s_tile[((sy + 1) * 18 + (sx + 1)) * 3 + c] + 2;\n"
"        r2 >>= 2;\n"
"\n"
"        // Vertical pass\n"
"        ushort res = (r0 + (r1 << 1) + r2 + 2) >> 2;\n"
"        out[(out_y * W + out_x) * 3 + c] = (uchar)res;\n"
"    }\n"
"}\n";

int main(int argc, char **argv) {
    @autoreleasepool {
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

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "Metal is not supported.\n");
            return 1;
        }
        printf("Metal Device: %s\n", [[device name] UTF8String]);

        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        NSError *error = nil;
        NSString *shaderSource = [NSString stringWithUTF8String:metal_shader_src];
        id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
        if (!library) {
            fprintf(stderr, "Shader compile error: %s\n", [[error localizedDescription] UTF8String]);
            return 1;
        }

        id<MTLFunction> kernelFunc = [library newFunctionWithName:@"conv_tiled_separable"];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:kernelFunc error:&error];
        if (!pipeline) {
            fprintf(stderr, "Pipeline error: %s\n", [[error localizedDescription] UTF8String]);
            return 1;
        }

        // True zero-copy: allocate page-aligned shared buffer
        id<MTLBuffer> inBuffer = [device newBufferWithBytes:in.data length:data_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> outBuffer = [device newBufferWithLength:data_size options:MTLResourceStorageModeShared];

        RegionParams params;
        params.x_start = 0;
        params.x_end = in.width;
        params.y_start = 0;
        params.y_end = in.height;
        params.img_width = in.width;
        params.img_height = in.height;
        params.channels = in.channels;

        // Warmup
        {
            id<MTLCommandBuffer> cb = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pipeline];
            [enc setBuffer:inBuffer offset:0 atIndex:0];
            [enc setBuffer:outBuffer offset:0 atIndex:1];
            [enc setBytes:&params length:sizeof(RegionParams) atIndex:2];
            [enc setThreadgroupMemoryLength:18 * 18 * 3 atIndex:0];

            MTLSize threadgroups = MTLSizeMake((in.width + 15) / 16, (in.height + 15) / 16, 1);
            MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
            [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadgroupSize];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }

        // Benchmark runs
        int ITERS = 10;
        double total_time = 0.0;
        for (int i = 0; i < ITERS; i++) {
            double t0 = get_time_sec();
            id<MTLCommandBuffer> cb = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pipeline];
            [enc setBuffer:inBuffer offset:0 atIndex:0];
            [enc setBuffer:outBuffer offset:0 atIndex:1];
            [enc setBytes:&params length:sizeof(RegionParams) atIndex:2];
            [enc setThreadgroupMemoryLength:18 * 18 * 3 atIndex:0];

            MTLSize threadgroups = MTLSizeMake((in.width + 15) / 16, (in.height + 15) / 16, 1);
            MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
            [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadgroupSize];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            total_time += get_time_sec() - t0;
        }

        double avg_time = total_time / ITERS;
        double mp_sec = ((double)in.width * in.height / 1e6) / avg_time;

        printf("\n--- Level 5: Metal Threadgroup-Tiled Separable Kernel ---\n");
        printf("Average Execution Time: %.6fs (%.3f ms)\n", avg_time, avg_time * 1000.0);
        printf("Throughput:             %.1f Megapixels/sec\n", mp_sec);

        // Verify bit-exact accuracy against CPU reference
        uint8_t *cpu_ref = (uint8_t *)malloc(data_size);
        // Quick reference on center row
        int mismatches = 0;
        int max_diff = 0;
        uint8_t *gpu_out = (uint8_t *)[outBuffer contents];

#define CLAMP_VAL(v, l, h) (((v) < (l)) ? (l) : (((v) > (h)) ? (h) : (v)))

        // Compute baseline 2D integer reference for comparison
        for (int y = 0; y < in.height; y++) {
            for (int x = 0; x < in.width; x++) {
                for (int c = 0; c < in.channels; c++) {
                    int p00 = in.data[(CLAMP_VAL(y-1, 0, in.height-1)*in.width + CLAMP_VAL(x-1, 0, in.width-1))*3 + c];
                    int p01 = in.data[(CLAMP_VAL(y-1, 0, in.height-1)*in.width + CLAMP_VAL(x,   0, in.width-1))*3 + c];
                    int p02 = in.data[(CLAMP_VAL(y-1, 0, in.height-1)*in.width + CLAMP_VAL(x+1, 0, in.width-1))*3 + c];
                    int p10 = in.data[(CLAMP_VAL(y,   0, in.height-1)*in.width + CLAMP_VAL(x-1, 0, in.width-1))*3 + c];
                    int p11 = in.data[(CLAMP_VAL(y,   0, in.height-1)*in.width + CLAMP_VAL(x,   0, in.width-1))*3 + c];
                    int p12 = in.data[(CLAMP_VAL(y,   0, in.height-1)*in.width + CLAMP_VAL(x+1, 0, in.width-1))*3 + c];
                    int p20 = in.data[(CLAMP_VAL(y+1, 0, in.height-1)*in.width + CLAMP_VAL(x-1, 0, in.width-1))*3 + c];
                    int p21 = in.data[(CLAMP_VAL(y+1, 0, in.height-1)*in.width + CLAMP_VAL(x,   0, in.width-1))*3 + c];
                    int p22 = in.data[(CLAMP_VAL(y+1, 0, in.height-1)*in.width + CLAMP_VAL(x+1, 0, in.width-1))*3 + c];

                    int r0 = (p00 + (p01 << 1) + p02 + 2) >> 2;
                    int r1 = (p10 + (p11 << 1) + p12 + 2) >> 2;
                    int r2 = (p20 + (p21 << 1) + p22 + 2) >> 2;
                    int expected = (r0 + (r1 << 1) + r2 + 2) >> 2;

                    int actual = gpu_out[(y * in.width + x) * 3 + c];
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

        free(cpu_ref);
        stbi_image_free(in.data);
    }
    return 0;
}
