#!/bin/zsh

qstat -F -f -q LUNG \
	| grep -E '@|mem_free|gpu.names|ngpus|/' \
	| grep -A 9 "^LUNG@argon-itf-bx55-04.hpc" \
	| grep -E --color=always '@argon-|=|seyhosseini' \
	| sed -n '1p;2p;4p;10p' \
	| (echo; cat;)

# qstat -j 547688 \
# 	| grep -E --color=always 'cpu=|mem=|io=|vmem=|maxvmem=' \
# 	| (cat;)

# nvidia-smi \
# 	| awk 'NR==14' \
# 	| grep -E --color=always 'MiB|/' \
# 	| (cat; echo)
