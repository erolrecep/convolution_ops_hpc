import sys
import time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from common import get_regions, KERNEL

def main():
    if len(sys.argv) < 4:
        print("Usage: pytorch_conv.py <technique> <input_image> <output_image>")
        sys.exit(1)

    technique = int(sys.argv[1])
    input_path = sys.argv[2]
    output_path = sys.argv[3]

    img = cv2.imread(input_path)
    if img is None:
        print(f"Error loading image {input_path}")
        sys.exit(1)
    H, W, C = img.shape
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Construct kernel weight tensor (C, 1, 3, 3) for depthwise conv2d
    kernel_tensor = torch.tensor(KERNEL, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    kernel_weights = kernel_tensor.repeat(C, 1, 1, 1)

    regions = get_regions(technique, W, H)
    out = np.zeros_like(img)

    # Pad image for boundary replication
    img_padded = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode='edge')
    # Transpose to (1, C, H_padded, W_padded) for PyTorch
    img_tensor = torch.tensor(img_padded, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)

    start_time = time.time()
    for x_start, x_end, y_start, y_end in regions:
        # Extract region with padding offsets
        sub_tensor = img_tensor[:, :, y_start:y_end+2, x_start:x_end+2]
        with torch.no_grad():
            res = F.conv2d(sub_tensor, kernel_weights, groups=C)
        res_np = res.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out[y_start:y_end, x_start:x_end] = np.clip(res_np + 0.5, 0, 255).astype(np.uint8)

    elapsed = time.time() - start_time

    if output_path != "none":
        cv2.imwrite(output_path, out)
    print(f"EXECUTION_TIME: {elapsed:.6f}")

if __name__ == "__main__":
    main()
