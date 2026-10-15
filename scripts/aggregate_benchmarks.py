#!/usr/bin/env python3
"""
aggregate_benchmarks.py
Aggregates benchmark results from all experiment folders across the 5 hardware targets:
- Apple M5 Max 16" (m5max16.local)
- AMD Threadripper 2950X + RTX 5060 Ti (dlbox)
- Raspberry Pi 5 16GB (rpi5ai)
- Raspberry Pi 4 8GB (rpi01)
- NVIDIA Jetson Nano 4GB (jetson4GB)

Outputs:
- Consolidated Markdown comparison table
- Consolidated CSV
- Cross-platform Gnuplot script & PNG charts
"""

import os
import sys
import glob
import csv
import subprocess

TARGET_HOSTS = ["m5max16.local", "dlbox", "jetson4GB", "rpi5ai", "rpi01"]

HOST_PRETTY_NAMES = {
    "m5max16.local": "Apple M5 Max 16\" (18C / 40GPU / 128GB)",
    "dlbox": "dlbox (Threadripper 2950X + RTX 5060 Ti)",
    "jetson4GB": "Jetson Nano (4C A57 + Maxwell 128-core)",
    "rpi5ai": "Raspberry Pi 5 (4C A76 @ 2.4GHz / 16GB)",
    "rpi01": "Raspberry Pi 4 (4C A72 @ 1.5GHz / 8GB)"
}

IMAGE_MEGAPIXELS = (7680 * 4320) / 1e6 # ~33.1776 MP

def find_latest_experiment_for_host(host):
    pattern = os.path.join("experiments", f"*_{host}")
    dirs = sorted(glob.glob(pattern))
    if not dirs:
        return None
    # Pick the most recent one that has a benchmark_results.csv
    for d in reversed(dirs):
        csv_file = os.path.join(d, "benchmark_results.csv")
        if os.path.exists(csv_file) and os.path.getsize(csv_file) > 100:
            return csv_file
    return None

def parse_csv(csv_path):
    results = {}
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) >= 3:
                target, tech, val = row[0].strip(), row[1].strip(), row[2].strip()
                results[(target, tech)] = val
    return results

def to_seconds(val_str):
    if not val_str or "N/A" in val_str or "Timeout" in val_str:
        return None
    try:
        clean = val_str.replace("s", "").strip()
        return float(clean)
    except ValueError:
        return None

def main():
    exp_data = {}
    found_hosts = []
    for host in TARGET_HOSTS:
        csv_path = find_latest_experiment_for_host(host)
        if csv_path:
            exp_data[host] = parse_csv(csv_path)
            found_hosts.append(host)
            print(f"Loaded {host} from {csv_path}")
        else:
            print(f"Warning: No valid benchmark_results.csv found for {host}")

    if not exp_data:
        print("No benchmark data available.")
        return

    # Extract all unique (target, technique) keys
    all_keys = []
    first_host = found_hosts[0]
    for key in exp_data[first_host].keys():
        if key not in all_keys:
            all_keys.append(key)

    # Generate Markdown Summary
    out_dir = os.path.join("experiments", "consolidated_summary")
    os.makedirs(out_dir, exist_ok=True)
    summary_md_path = os.path.join(out_dir, "hardware_comparison.md")
    summary_csv_path = os.path.join(out_dir, "hardware_comparison.csv")

    with open(summary_md_path, "w") as f_md, open(summary_csv_path, "w", newline="") as f_csv:
        csv_writer = csv.writer(f_csv)
        header = ["Target", "Technique"] + [HOST_PRETTY_NAMES.get(h, h) for h in found_hosts]
        csv_writer.writerow(header)

        f_md.write("# 8K Convolution Multi-Hardware HPC Benchmark Comparison\n\n")
        f_md.write(f"**Input Image**: BMW M3 Touring 24h (7680x4320, {IMAGE_MEGAPIXELS:.2f} Megapixels, 3 channels)\n\n")
        f_md.write("### Measured Wall-Clock Times (seconds)\n\n")

        # Table header
        f_md.write("| Target | Technique | " + " | ".join([HOST_PRETTY_NAMES.get(h, h) for h in found_hosts]) + " |\n")
        f_md.write("| :--- | :--- | " + " | ".join([":---:" for _ in found_hosts]) + " |\n")

        current_target = None
        for target, tech in all_keys:
            row = [target, tech]
            md_cells = []
            for h in found_hosts:
                val = exp_data[h].get((target, tech), "N/A")
                row.append(val)
                sec = to_seconds(val)
                if sec is not None:
                    md_cells.append(f"**{sec:.4f}s**" if sec < 0.1 else f"{sec:.4f}s")
                elif "Timeout" in val:
                    md_cells.append("*(Timeout >45s)*")
                else:
                    md_cells.append("—")
            csv_writer.writerow(row)
            f_md.write(f"| {target} | {tech} | " + " | ".join(md_cells) + " |\n")

        # Write Peak Performance and Speedup Summary
        f_md.write("\n\n### Peak Hardware Throughput & Speedup Summary\n\n")
        f_md.write("| Device | Fastest Target | Best Time (s) | Throughput (MPixels/s) | Speedup vs Single-Thread C |\n")
        f_md.write("| :--- | :--- | :---: | :---: | :---: |\n")

        for h in found_hosts:
            data = exp_data[h]
            best_time = float("inf")
            best_target = None
            pure_c_time = to_seconds(data.get(("Pure C", "Technique 1"))) or 1.0

            for (target, tech), val in data.items():
                s = to_seconds(val)
                if s is not None and s < best_time:
                    best_time = s
                    best_target = f"{target} ({tech})"

            if best_time < float("inf"):
                throughput = IMAGE_MEGAPIXELS / best_time
                speedup = pure_c_time / best_time
                f_md.write(f"| **{HOST_PRETTY_NAMES.get(h, h)}** | {best_target} | **{best_time:.4f}s** | **{throughput:.1f} MP/s** | **{speedup:.2f}x** |\n")

    print(f"Wrote consolidated Markdown table to: {summary_md_path}")
    print(f"Wrote consolidated CSV to: {summary_csv_path}")

    # Generate Gnuplot Cross-Platform Charts
    gnuplot_script = os.path.join(out_dir, "plot_comparison.gp")
    data_file = os.path.join(out_dir, "best_times.dat")

    with open(data_file, "w") as f_dat:
        f_dat.write("# Device PureC_s Pthreads_s OpenMP_s GPU_s BestPython_s\n")
        for h in found_hosts:
            d = exp_data[h]
            pure_c = min([to_seconds(d.get(("Pure C", f"Technique {t}"))) or 999.0 for t in range(1, 5)])
            pthreads = min([to_seconds(d.get(("C with pthreads", f"Technique {t}"))) or 999.0 for t in range(1, 5)])
            openmp = min([to_seconds(d.get(("C with openMP", f"Technique {t}"))) or 999.0 for t in range(1, 5)])
            
            # GPU: Metal on Apple, CUDA on NVIDIA
            gpu_times = []
            for t in range(1, 5):
                m = to_seconds(d.get(("C with Metal", f"Technique {t}")))
                c = to_seconds(d.get(("C with CUDA", f"Technique {t}")))
                if m: gpu_times.append(m)
                if c: gpu_times.append(c)
            gpu = min(gpu_times) if gpu_times else 999.0

            # Best python
            py_times = []
            for p_target in ["Python with OpenCV", "Python with Numba", "Python with NumPy", "Python with PyTorch"]:
                for t in range(1, 5):
                    s = to_seconds(d.get((p_target, f"Technique {t}")))
                    if s: py_times.append(s)
            best_py = min(py_times) if py_times else 999.0

            short_name = h.split(".")[0]
            f_dat.write(f'"{short_name}" {pure_c:.4f} {pthreads:.4f} {openmp:.4f} {gpu:.4f} {best_py:.4f}\n')

    with open(gnuplot_script, "w") as f_gp:
        f_gp.write("""
set terminal pngcairo size 1200,750 enhanced font 'Arial,12'
set output 'experiments/consolidated_summary/device_comparison.png'

set title '8K Convolution Execution Time Across Architectures (Log Scale)' font ',14'
set style data histogram
set style histogram cluster gap 1
set style fill solid 0.8 border -1
set boxwidth 0.85
set logscale y
set format y "10^{%T}s"
set grid ytics lc rgb '#dcdcdc'
set key top right outside

set xlabel "Hardware Platform"
set ylabel "Execution Time (seconds, log scale)"

plot 'experiments/consolidated_summary/best_times.dat' using 2:xtic(1) title 'Pure C (Single Core)' lc rgb '#e41a1c', \
     '' using 3 title 'C pthreads' lc rgb '#377eb8', \
     '' using 4 title 'C OpenMP' lc rgb '#4daf4a', \
     '' using 5 title 'Hardware Accelerator (CUDA/Metal)' lc rgb '#984ea3', \
     '' using 6 title 'Best Python (OpenCV/Numba)' lc rgb '#ff7f00'
""")

    # Try running gnuplot
    try:
        subprocess.run(["gnuplot", gnuplot_script], check=True)
        print("Generated cross-platform comparison chart at: experiments/consolidated_summary/device_comparison.png")
    except Exception as e:
        print(f"Gnuplot execution note: {e}")

if __name__ == "__main__":
    main()
