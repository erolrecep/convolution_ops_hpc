#include "../include/common.h"

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
    out.data = (unsigned char *)malloc(in.width * in.height * in.channels);

    int num_regions = 0;
    Region *regions = get_regions(technique, in.width, in.height, &num_regions);

    double start_time = get_time_sec();
    for (int i = 0; i < num_regions; i++) {
        apply_convolution_region(&in, &out, regions[i]);
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
