import sys
import time
import numpy as np
import cv2
from common import get_regions, KERNEL

def apply_region_opencv(img, out, x_start, x_end, y_start, y_end):
    H, W, _ = img.shape
    # Add 1px border around region for boundary filtering
    r_y_start = max(0, y_start - 1)
    r_y_end = min(H, y_end + 1)
    r_x_start = max(0, x_start - 1)
    r_x_end = min(W, x_end + 1)

    sub_img = img[r_y_start:r_y_end, r_x_start:r_x_end]
    filtered = cv2.filter2D(sub_img, -1, KERNEL, borderType=cv2.BORDER_REPLICATE)

    # Offset to target region
    off_y_start = y_start - r_y_start
    off_y_end = off_y_start + (y_end - y_start)
    off_x_start = x_start - r_x_start
    off_x_end = off_x_start + (x_end - x_start)

    out[y_start:y_end, x_start:x_end] = filtered[off_y_start:off_y_end, off_x_start:off_x_end]

def main():
    if len(sys.argv) < 4:
        print("Usage: opencv_conv.py <technique> <input_image> <output_image>")
        sys.exit(1)

    technique = int(sys.argv[1])
    input_path = sys.argv[2]
    output_path = sys.argv[3]

    img = cv2.imread(input_path)
    if img is None:
        print(f"Error loading image {input_path}")
        sys.exit(1)

    H, W, _ = img.shape
    out = np.zeros_like(img)

    regions = get_regions(technique, W, H)

    start_time = time.time()
    for x_start, x_end, y_start, y_end in regions:
        apply_region_opencv(img, out, x_start, x_end, y_start, y_end)
    elapsed = time.time() - start_time

    if output_path != "none":
        cv2.imwrite(output_path, out)
    print(f"EXECUTION_TIME: {elapsed:.6f}")

if __name__ == "__main__":
    main()
