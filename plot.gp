set xdata time
set timefmt "%H:%M:%S"
set format x "%H:%M:%S"
set xlabel "Time"
set ylabel "Free Memory (MB)"
set title "Free Memory Monitoring"
plot "memory.log" using 1:2 with lines title "Free Memory"
