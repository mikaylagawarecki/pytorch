#!/usr/bin/bash
# Usage: ./run_bench_8b.sh {flags to pass to bench_llama_8b.py}
# See docstring at top of bench_llama_8b.py for definition of flags

counter=0
max_counter=10

while [ $counter -lt $max_counter ]; do
  echo "Iteration $counter: Running generate_new_checkpoint.py -s 8"
  python generate_new_checkpoint.py -s 8
  echo "Iteration $counter: Running bench_llama_8b.py $@"
  torchrun --standalone --nproc-per-node 1 bench_llama_8b.py "$@"
  ((counter++))
done