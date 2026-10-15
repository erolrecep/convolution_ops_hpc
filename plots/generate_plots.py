import os
import csv
import sys
import subprocess

def parse_csv(csv_path):
    data = {}
    if not os.path.exists(csv_path):
        return data

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            target = row["Target"]
            tech = row["Technique"]
            time_raw = row["Time"]
            if time_raw.endswith("s"):
                time_raw = time_raw[:-1]
            try:
                val = float(time_raw)
            except ValueError:
                val = 0.0
            if target not in data:
                data[target] = {}
            data[target][tech] = val
    return data

def main():
    # Accept custom CSV path and output folder arguments
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results.csv"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "."

    data = parse_csv(csv_path)

    os.makedirs(output_dir, exist_ok=True)
    dat_path = os.path.join(output_dir, "benchmark_data.dat")

    # Write formatted data file for gnuplot
    targets = [
        "Pure C", "C with pthreads", "C with openMP", "C with CUDA", "C with Metal",
        "Pure Python", "Python with NumPy", "Python with OpenCV",
        "Python with Numba", "Python with CuPy", "Python with PyTorch"
    ]

    with open(dat_path, "w") as f:
        f.write("# Target\tTech1\tTech2\tTech3\tTech4\n")
        for t in targets:
            if t in data:
                t1 = data[t].get("Technique 1", 0.0)
                t2 = data[t].get("Technique 2", 0.0)
                t3 = data[t].get("Technique 3", 0.0)
                t4 = data[t].get("Technique 4", 0.0)
                f.write(f'"{t}"\t{t1:.4f}\t{t2:.4f}\t{t3:.4f}\t{t4:.4f}\n')

    # Path helpers for outputs
    bar_out = os.path.join(output_dir, "bar_graph.png")
    line_out = os.path.join(output_dir, "line_graph.png")
    scatter_out = os.path.join(output_dir, "scatter_plot.png")

    # 1. Bar Graph script content
    bar_gp = f"""
set terminal pngcairo size 1200,800 font 'Helvetica,12'
set output '{bar_out}'
set title 'Convolution Benchmark Execution Time (Bar Graph)' font 'Helvetica-Bold,16'
set style data histogram
set style histogram cluster gap 1
set style fill solid 0.8 border -1
set boxwidth 0.9
set xtics rotate by -45 font ',10'
set grid y
set ylabel 'Time (seconds)'
plot '{dat_path}' using 2:xtic(1) title 'Technique 1', \
     '' using 3 title 'Technique 2', \
     '' using 4 title 'Technique 3', \
     '' using 5 title 'Technique 4'
"""

    # 2. Line Graph script content
    line_gp = f"""
set terminal pngcairo size 1200,800 font 'Helvetica,12'
set output '{line_out}'
set title 'Convolution Benchmark Execution Time Trends (Line Graph)' font 'Helvetica-Bold,16'
set key right top
set grid
set xlabel 'Technique'
set ylabel 'Time (seconds)'
set xtics ('Tech 1' 1, 'Tech 2' 2, 'Tech 3' 3, 'Tech 4' 4)
plot '{dat_path}' using (1):2 with linespoints lw 2 pt 7 title 'Tech 1', \
     '{dat_path}' using (2):3 with linespoints lw 2 pt 9 title 'Tech 2', \
     '{dat_path}' using (3):4 with linespoints lw 2 pt 11 title 'Tech 3', \
     '{dat_path}' using (4):5 with linespoints lw 2 pt 13 title 'Tech 4'
"""

    # 3. Scatter Plot script content
    scatter_gp = f"""
set terminal pngcairo size 1200,800 font 'Helvetica,12'
set output '{scatter_out}'
set title 'Convolution Benchmark Execution Time Scatter Distribution' font 'Helvetica-Bold,16'
set key right top
set grid
set xlabel 'Implementation Target'
set ylabel 'Time (seconds)'
set xtics rotate by -45 font ',10'
plot '{dat_path}' using 0:2:xtic(1) with points pt 7 ps 2 title 'Technique 1', \
     '' using 0:3 with points pt 9 ps 2 title 'Technique 2', \
     '' using 0:4 with points pt 11 ps 2 title 'Technique 3', \
     '' using 0:5 with points pt 13 ps 2 title 'Technique 4'
"""

    import shutil
    gnuplot_cmd = shutil.which("gnuplot")
    if not gnuplot_cmd:
        for candidate in [
            os.path.expanduser("~/miniforge3/bin/gnuplot"),
            os.path.join(os.path.dirname(sys.executable), "gnuplot"),
            "/usr/bin/gnuplot", "/usr/local/bin/gnuplot", "/opt/homebrew/bin/gnuplot"
        ]:
            if os.path.exists(candidate):
                gnuplot_cmd = candidate
                break
    if not gnuplot_cmd:
        gnuplot_cmd = "gnuplot"

    # Execute gnuplot commands by piping the commands directly to gnuplot stdin
    for gp_content in [bar_gp, line_gp, scatter_gp]:
        p = subprocess.Popen([gnuplot_cmd], stdin=subprocess.PIPE, universal_newlines=True)
        p.communicate(input=gp_content)

    print(f"Generated performance graphs in: {output_dir}")

if __name__ == "__main__":
    main()
