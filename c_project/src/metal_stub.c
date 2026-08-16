#include "../include/common.h"

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <technique_num 1-4> <input_image> <output_image>\n", argv[0]);
        return 1;
    }

    // Metal is only supported on macOS / Apple Silicon hardware.
    printf("EXECUTION_TIME: N/A\n");
    return 0;
}
