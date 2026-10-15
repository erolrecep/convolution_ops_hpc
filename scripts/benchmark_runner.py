import os
import sys
import time
import subprocess
import csv
from datetime import datetime

TARGETS = [
    ("Pure C", ["./c_project/bin/pure_c"]),
    ("C with pthreads", ["./c_project/bin/pthreads_c"]),
    ("C with openMP", ["./c_project/bin/openmp_c"]),
    ("C with CUDA", ["./c_project/bin/cuda_c"]),
    ("C with Metal", ["./c_project/bin/metal_c"]),
    ("Pure Python", [sys.executable, "python_project/run_python.py", "pure_python"]),
    ("Python with NumPy", [sys.executable, "python_project/run_python.py", "numpy"]),
    ("Python with OpenCV", [sys.executable, "python_project/run_python.py", "opencv"]),
    ("Python with Numba", [sys.executable, "python_project/run_python.py", "numba"]),
    ("Python with CuPy", [sys.executable, "python_project/run_python.py", "cupy"]),
    ("Python with PyTorch", [sys.executable, "python_project/run_python.py", "pytorch"]),
]

def run_single(cmd_base, tech, input_img, out_img):
    cmd = cmd_base + [str(tech), input_img, out_img]
    try:
        timeout_val = 45 if "pure_python" in cmd_base else 30
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=timeout_val)
        if res.returncode != 0:
            return "N/A"
        for line in res.stdout.splitlines():
            if "EXECUTION_TIME:" in line:
                val = line.split("EXECUTION_TIME:")[1].strip()
                if val == "N/A":
                    return "N/A"
                try:
                    fval = float(val)
                    return f"{fval:.4f}s"
                except ValueError:
                    return "N/A"
        return "N/A"
    except subprocess.TimeoutExpired:
        try:
            subprocess.run(["pkill", "-9", "-f", "pure_python.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
        return "Timeout (>45s)"
    except Exception as e:
        return "N/A"

import socket

def check_cuda():
    if os.path.exists("/usr/local/cuda/bin") and "/usr/local/cuda/bin" not in os.environ.get("PATH", ""):
        os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")
    has_nvcc = subprocess.run(["which", "nvcc"], stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0
    return has_nvcc

def main():
    if len(sys.argv) < 2:
        print("Usage: benchmark_runner.py <input_image>")
        sys.exit(1)

    input_img = sys.argv[1]
    if not os.path.exists(input_img):
        print(f"Image file not found: {input_img}")
        sys.exit(1)

    # Detect hardware capabilities
    has_cuda = check_cuda()

    # Determine active targets
    active_targets = []
    for name, cmd in TARGETS:
        if "CUDA" in name or "CuPy" in name:
            if not has_cuda:
                continue
        active_targets.append((name, cmd))

    # Total runs: active targets x 4 techniques
    total_runs = len(active_targets) * 4

    # Timestamp, hostname and folder setup
    hostname = socket.gethostname()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join("experiments", f"{timestamp}_{hostname}")
    results_dir = os.path.join(exp_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Create experiment.yaml
    yaml_path = os.path.join(exp_dir, "experiment.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"hostname: \"{hostname}\"\n")
        f.write(f"input_image: \"{input_img}\"\n")
        f.write(f"total_runs: {total_runs}\n")
        f.write("experiments:\n")
        for name, _ in active_targets:
            for tech in range(1, 5):
                f.write(f"  - target: \"{name}\"\n")
                f.write(f"    technique: {tech}\n")

    print(f"Created experiment setup directory: {exp_dir}")
    print(f"Saved {yaml_path}")

    # Run experiments
    results = {}
    csv_rows = []

    # Initialize results structure
    for name, _ in TARGETS:
        results[name] = {1: "N/A", 2: "N/A", 3: "N/A", 4: "N/A"}

    for name, cmd in TARGETS:
        # If hardware does not support CUDA, we record N/A directly
        if ("CUDA" in name or "CuPy" in name) and not has_cuda:
            for tech in range(1, 5):
                csv_rows.append({
                    "Target": name,
                    "Technique": f"Technique {tech}",
                    "Time": "N/A"
                })
            continue

        for tech in range(1, 5):
            print(f"Running [{name}] Technique {tech}...")
            # Do not record the output image to folder, pass "none"
            time_str = run_single(cmd, tech, input_img, "none")
            results[name][tech] = time_str
            print(f"  -> Result: {time_str}")
            csv_rows.append({
                "Target": name,
                "Technique": f"Technique {tech}",
                "Time": time_str
            })

    # Save CSV inside the experiment run folder
    csv_path = os.path.join(exp_dir, "benchmark_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Target", "Technique", "Time"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved results to {csv_path}")

    # Render plots to results subfolder
    print("Generating gnuplot performance graphs...")
    subprocess.run([sys.executable, "plots/generate_plots.py", csv_path, results_dir])

    # Update global benchmark_results.csv for top-level compat
    with open("benchmark_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Target", "Technique", "Time"])
        writer.writeheader()
        writer.writerows(csv_rows)

    # Generate markdown table
    table_lines = [
        "| |Technique 1|Technique 2|Technique 3|Technique 4|",
        "|--|--|--|--|--|"
    ]
    for name, _ in TARGETS:
        t1 = results[name][1]
        t2 = results[name][2]
        t3 = results[name][3]
        t4 = results[name][4]
        table_lines.append(f"|**{name}**| {t1} | {t2} | {t3} | {t4} |")

    new_table_str = "\n".join(table_lines) + "\n"

    # Update README.md
    readme_path = "README.md"
    if os.path.exists(readme_path):
        with open(readme_path, "r") as f:
            content = f.read()

        start_marker = "## Expected Benchmark Table"
        todo_marker = "## To-Do List"

        if start_marker in content and todo_marker in content:
            pre = content.split(start_marker)[0] + start_marker + "\n\nBuild a 2D table on one-side is programming language with tools and other size is the programming technique. \n\n - Rows are the programming languages with tools.\n - Columns are the programming techniques.\n\n"
            post = "\n\n" + todo_marker + content.split(todo_marker)[1]
            updated_content = pre + new_table_str + post
            with open(readme_path, "w") as f:
                f.write(updated_content)
            print("Successfully updated README.md table with benchmark results!")

if __name__ == "__main__":
    main()
