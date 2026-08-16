#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../include/common.h"

typedef struct {
    uint32_t x_start;
    uint32_t x_end;
    uint32_t y_start;
    uint32_t y_end;
    uint32_t img_width;
    uint32_t img_height;
    uint32_t channels;
} MetalRegionParams;

static const char *metal_shader_src =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct RegionParams {\n"
"    uint x_start;\n"
"    uint x_end;\n"
"    uint y_start;\n"
"    uint y_end;\n"
"    uint img_width;\n"
"    uint img_height;\n"
"    uint channels;\n"
"};\n"
"constant float g_kernel[9] = {\n"
"    1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f,\n"
"    2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f,\n"
"    1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f\n"
"};\n"
"kernel void convolution_kernel(\n"
"    device const uchar* input_data [[buffer(0)]],\n"
"    device uchar* output_data [[buffer(1)]],\n"
"    constant RegionParams& reg [[buffer(2)]],\n"
"    uint2 thread_position [[thread_position_in_grid]])\n"
"{\n"
"    uint x = reg.x_start + thread_position.x;\n"
"    uint y = reg.y_start + thread_position.y;\n"
"    if (x >= reg.x_end || y >= reg.y_end) return;\n"
"    int W = reg.img_width;\n"
"    int H = reg.img_height;\n"
"    int ch = reg.channels;\n"
"    for (int c = 0; c < ch; c++) {\n"
"        float sum = 0.0f;\n"
"        for (int ky = -1; ky <= 1; ky++) {\n"
"            for (int kx = -1; kx <= 1; kx++) {\n"
"                int py = (int)y + ky;\n"
"                int px = (int)x + kx;\n"
"                if (py < 0) py = 0; if (py >= H) py = H - 1;\n"
"                if (px < 0) px = 0; if (px >= W) px = W - 1;\n"
"                float k_val = g_kernel[(ky + 1) * 3 + (kx + 1)];\n"
"                sum += (float)input_data[(py * W + px) * ch + c] * k_val;\n"
"            }\n"
"        }\n"
"        output_data[(y * W + x) * ch + c] = (uchar)(sum < 0.0f ? 0.0f : (sum > 255.0f ? 255.0f : (sum + 0.5f)));\n"
"    }\n"
"}\n";

int main(int argc, char **argv) {
    @autoreleasepool {
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

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "Metal is not supported on this system.\n");
            return 1;
        }

        id<MTLCommandQueue> commandQueue = [device newCommandQueue];

        NSError *error = nil;
        NSString *shaderSource = [NSString stringWithUTF8String:metal_shader_src];
        id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
        if (!library) {
            fprintf(stderr, "Failed to compile Metal shader: %s\n", [[error localizedDescription] UTF8String]);
            return 1;
        }

        id<MTLFunction> kernelFunc = [library newFunctionWithName:@"convolution_kernel"];
        id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:kernelFunc error:&error];
        if (!pipelineState) {
            fprintf(stderr, "Failed to create Metal pipeline state: %s\n", [[error localizedDescription] UTF8String]);
            return 1;
        }

        id<MTLBuffer> inBuffer = [device newBufferWithBytes:in.data length:data_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> outBuffer = [device newBufferWithLength:data_size options:MTLResourceStorageModeShared];

        int num_regions = 0;
        Region *regions = get_regions(technique, in.width, in.height, &num_regions);

        double start_time = get_time_sec();

        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:pipelineState];
        [encoder setBuffer:inBuffer offset:0 atIndex:0];
        [encoder setBuffer:outBuffer offset:0 atIndex:1];

        for (int i = 0; i < num_regions; i++) {
            MetalRegionParams params;
            params.x_start = regions[i].x_start;
            params.x_end = regions[i].x_end;
            params.y_start = regions[i].y_start;
            params.y_end = regions[i].y_end;
            params.img_width = in.width;
            params.img_height = in.height;
            params.channels = in.channels;

            [encoder setBytes:&params length:sizeof(MetalRegionParams) atIndex:2];

            MTLSize gridSize = MTLSizeMake(params.x_end - params.x_start, params.y_end - params.y_start, 1);
            NSUInteger threadGroupWidth = pipelineState.maxTotalThreadsPerThreadgroup;
            if (threadGroupWidth > 16) threadGroupWidth = 16;
            MTLSize threadGroupSize = MTLSizeMake(threadGroupWidth, 16, 1);

            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        }

        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        double elapsed = get_time_sec() - start_time;

        memcpy(out.data, [outBuffer contents], data_size);
        if (strcmp(output_path, "none") != 0) {
            stbi_write_jpg(output_path, out.width, out.height, out.channels, out.data, 90);
        }

        free(regions);
        stbi_image_free(in.data);
        free(out.data);

        printf("EXECUTION_TIME: %.6f\n", elapsed);
    }
    return 0;
}
