#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../../c_project/include/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../c_project/include/stb_image_write.h"

int c_get_image_dims(const char *path, int *w, int *h, int *ch) {
    return stbi_info(path, w, h, ch);
}

int c_load_image_to_ptr(const char *path, unsigned char *dst) {
    int w, h, ch;
    unsigned char *data = stbi_load(path, &w, &h, &ch, 0);
    if (!data) return 0;
    memcpy(dst, data, (size_t)w * h * ch);
    stbi_image_free(data);
    return 1;
}

void c_save_image_from_ptr(const char *path, int w, int h, int ch, const unsigned char *src) {
    if (path && strcmp(path, "none") != 0) {
        stbi_write_jpg(path, w, h, ch, src, 90);
    }
}

double c_get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
