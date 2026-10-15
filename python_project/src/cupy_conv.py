import sys
import time
import cv2
import numpy as np

from common import get_regions, KERNEL

def main():
    if len(sys.argv) < 4:
        print("Usage: cupy_conv.py <technique> <input_image> <output_image>")
        sys.exit(1)

    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() == 0:
            raise RuntimeError("No CUDA devices found")
    except Exception:
        print("EXECUTION_TIME: N/A")
        return

    technique = int(sys.argv[1])
    input_path = sys.argv[2]
    output_path = sys.argv[3]

    img = cv2.imread(input_path)
    if img is None:
        print(f"Error loading image {input_path}")
        sys.exit(1)

    H, W, C = img.shape
    padded = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode='edge').astype(np.float32)

    d_padded = cp.asarray(padded)
    d_kernel = cp.asarray(KERNEL, dtype=cp.float32)
    d_out = cp.zeros((H, W, C), dtype=cp.uint8)

    regions = get_regions(technique, W, H)

    cp.cuda.Stream.null.synchronize()
    start_time = time.time()

    for x_start, x_end, y_start, y_end in regions:
        sub_out = cp.zeros((y_end - y_start, x_end - x_start, C), dtype=cp.float32)
        for ky in range(3):
            for kx in range(3):
                k_val = d_kernel[ky, kx]
                slice_y = d_padded[y_start + ky : y_end + ky, x_start + kx : x_end + kx, :]
                sub_out += slice_y * k_val
        d_out[y_start:y_end, x_start:x_end] = cp.clip(sub_out + 0.5, 0, 255).astype(cp.uint8)

    cp.cuda.Stream.null.synchronize()
    elapsed = time.time() - start_time

    out = cp.asnumpy(d_out)
    if output_path != "none":
        cv2.imwrite(output_path, out)
    print(f"EXECUTION_TIME: {elapsed:.6f}")

if __name__ == "__main__":
    main()
