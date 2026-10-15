set terminal pngcairo size 1350,800 enhanced font 'Arial,12'
set output 'experiments/consolidated_summary/device_comparison.png'

set title '8K Convolution Multi-Language & Architecture Benchmark (Log Scale)' font ',15'
set style data histogram
set style histogram cluster gap 1
set style fill solid 0.85 border -1
set boxwidth 0.90
set logscale y
set yrange [0.0008:30]
set format y "10^{%T}s"
set grid ytics lc rgb '#dcdcdc'
set key top right outside font ',10'

set xlabel "Physical Hardware Platform" offset 0,-0.5 font ',12'
set ylabel "Execution Time (seconds, log scale)" font ',12'

plot 'experiments/consolidated_summary/best_times.dat' \
     using 2:xtic(1) title 'Pure C (1-Core)' lc rgb '#e41a1c', \
     '' using 3 title 'Fortran OpenMP' lc rgb '#4daf4a', \
     '' using 4 title 'Common Lisp (SBCL)' lc rgb '#984ea3', \
     '' using 5 title 'Go (Goroutines)' lc rgb '#00bcd4', \
     '' using 6 title 'Mojo 1.1' lc rgb '#ff7f00', \
     '' using 7 title 'C Level 4 Fused Ring-Buffer' lc rgb '#377eb8', \
     '' using 8 title 'Level 5 GPU Tiled (CUDA/Metal)' lc rgb '#2ca02c'
