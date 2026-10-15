
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

plot 'experiments/consolidated_summary/best_times.dat' using 2:xtic(1) title 'Pure C (Single Core)' lc rgb '#e41a1c',      '' using 3 title 'C pthreads' lc rgb '#377eb8',      '' using 4 title 'C OpenMP' lc rgb '#4daf4a',      '' using 5 title 'Hardware Accelerator (CUDA/Metal)' lc rgb '#984ea3',      '' using 6 title 'Best Python (OpenCV/Numba)' lc rgb '#ff7f00'
