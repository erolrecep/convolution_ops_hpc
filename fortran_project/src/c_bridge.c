#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../../c_project/include/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../c_project/include/stb_image_write.h"

unsigned char* load_image_c(const char* filename, int* width, int* height, int* channels) {
    return stbi_load(filename, width, height, channels, 0);
}

void free_image_c(unsigned char* data) {
    if (data) stbi_image_free(data);
}

void save_image_c(const char* filename, int width, int height, int channels, const unsigned char* data) {
    if (filename && strcmp(filename, "none") != 0) {
        stbi_write_jpg(filename, width, height, channels, data, 90);
    }
}

double get_wall_time_c(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
