import sys
import time
import numpy as np
import cv2
import numba
from numba import njit, prange
from common import get_regions

@njit(parallel=True, fastmath=True)
def numba_conv_kernel(img, out, x_start, x_end, y_start, y_end):
    H, W, C = img.shape
    kernel = np.array([
        [1.0/16.0, 2.0/16.0, 1.0/16.0],
        [2.0/16.0, 4.0/16.0, 2.0/16.0],
        [1.0/16.0, 2.0/16.0, 1.0/16.0]
    ], dtype=np.float32)

    for y in prange(y_start, y_end):
        for x in range(x_start, x_end):
            for c in range(C):
                val = 0.0
                for ky in range(-1, 2):
                    for kx in range(-1, 2):
                        py = max(0, min(H - 1, y + ky))
                        px = max(0, min(W - 1, x + kx))
                        val += float(img[py, px, c]) * kernel[ky + 1, kx + 1]
                out[y, x, c] = np.uint8(min(max(val + 0.5, 0.0), 255.0))

def main():
    if len(sys.argv) < 4:
        print("Usage: numba_conv.py <technique> <input_image> <output_image>")
        sys.exit(1)

    technique = int(sys.argv[1])
    input_path = sys.argv[2]
    output_path = sys.argv[3]

    img = cv2.imread(input_path)
    if img is None:
        print(f"Error loading image {input_path}")
        sys.exit(1)

    H, W, C = img.shape
    out = np.zeros_like(img)

    regions = get_regions(technique, W, H)

    # Warm-up JIT compilation on a tiny region
    numba_conv_kernel(img[:4, :4], out[:4, :4], 0, 4, 0, 4)

    start_time = time.time()
    for x_start, x_end, y_start, y_end in regions:
        numba_conv_kernel(img, out, x_start, x_end, y_start, y_end)
    elapsed = time.time() - start_time

    if output_path != "none":
        cv2.imwrite(output_path, out)
    print(f"EXECUTION_TIME: {elapsed:.6f}")

if __name__ == "__main__":
    main()
