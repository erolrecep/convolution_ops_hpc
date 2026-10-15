import os
import math
import numpy as np

KERNEL = np.array([
    [1.0/16.0, 2.0/16.0, 1.0/16.0],
    [2.0/16.0, 4.0/16.0, 2.0/16.0],
    [1.0/16.0, 2.0/16.0, 1.0/16.0]
], dtype=np.float32)

def get_num_cores():
    return os.cpu_count() or 4

def get_grid_dimensions(n):
    best_r, best_c = 1, n
    for r in range(1, int(n**0.5) + 1):
        if n % r == 0:
            best_r = r
            best_c = n // r
    return best_r, best_c

def get_regions(technique, width, height):
    n = get_num_cores()
    regions = []

    if technique == 1:
        R, C = get_grid_dimensions(n)
        for r in range(R):
            for c in range(C):
                y_start = r * (height // R)
                y_end = height if r == R - 1 else (r + 1) * (height // R)
                x_start = c * (width // C)
                x_end = width if c == C - 1 else (c + 1) * (width // C)
                regions.append((x_start, x_end, y_start, y_end))

    elif technique == 2:
        # Region 1: Top-left corner (1x1)
        regions.append((0, 1, 0, 1))
        # Region 2: Top edge (width-2 x 1)
        regions.append((1, width - 1, 0, 1))
        # Region 3: Top-right corner (1x1)
        regions.append((width - 1, width, 0, 1))
        # Region 4: Left edge (1 x height-2)
        regions.append((0, 1, 1, height - 1))
        # Region 5: Center region (width-2 x height-2)
        regions.append((1, width - 1, 1, height - 1))
        # Region 6: Right edge (1 x height-2)
        regions.append((width - 1, width, 1, height - 1))
        # Region 7: Bottom-left corner (1x1)
        regions.append((0, 1, height - 1, height))
        # Region 8: Bottom edge (width-2 x 1)
        regions.append((1, width - 1, height - 1, height))
        # Region 9: Bottom-right corner (1x1)
        regions.append((width - 1, width, height - 1, height))

    elif technique == 3:
        for r in range(n):
            y_start = r * (height // n)
            y_end = height if r == n - 1 else (r + 1) * (height // n)
            regions.append((0, width, y_start, y_end))

    elif technique == 4:
        for c in range(n):
            x_start = c * (width // n)
            x_end = width if c == n - 1 else (c + 1) * (width // n)
            regions.append((x_start, x_end, 0, height))

    return regions
