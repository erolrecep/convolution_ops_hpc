import sys
import time
import numpy as np
import cv2
from common import get_regions, KERNEL

def apply_region_numpy(padded_img, out, x_start, x_end, y_start, y_end):
    C = padded_img.shape[2]
    sub_out = np.zeros((y_end - y_start, x_end - x_start, C), dtype=np.float32)
    for ky in range(3):
        for kx in range(3):
            k_val = KERNEL[ky, kx]
            slice_y = padded_img[y_start + ky : y_end + ky, x_start + kx : x_end + kx, :]
            sub_out += slice_y * k_val
    out[y_start:y_end, x_start:x_end] = np.clip(sub_out + 0.5, 0, 255).astype(np.uint8)

def main():
    if len(sys.argv) < 4:
        print("Usage: numpy_conv.py <technique> <input_image> <output_image>")
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
    padded = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode='edge').astype(np.float32)

    regions = get_regions(technique, W, H)

    start_time = time.time()
    for x_start, x_end, y_start, y_end in regions:
        apply_region_numpy(padded, out, x_start, x_end, y_start, y_end)
    elapsed = time.time() - start_time

    if output_path != "none":
        cv2.imwrite(output_path, out)
    print(f"EXECUTION_TIME: {elapsed:.6f}")

if __name__ == "__main__":
    main()
