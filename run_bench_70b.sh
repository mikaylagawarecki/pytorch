#!/usr/bin/bash

# Get the value of the --num_ranks flag
if [ "$1" != "--num_ranks" ]; then
  echo "Usage ./run_bench_70b.sh --num_ranks {1/2/8} {flags to pass to bench_llama_70b.py}"
  echo "See docstring at top of bench_llama_70b.py for definition of flags"
  exit 1
else
  NUM_RANKS=$2
  counter=0
fi

shift
shift

max_counter=10

if [ $NUM_RANKS = 8 ]; then
  while [ $counter -lt $max_counter ]; do
    echo "Iteration $counter: Running generate_new_checkpoint.py -s 70"
    torchrun --standalone --nproc-per-node 8 generate_new_checkpoint.py -s 70
    echo "Iteration $counter: Running bench_llama_70b.py $@"
    torchrun --standalone --nproc-per-node 8 bench_llama_70b.py "$@"
    ((counter++))
  done
elif [ $NUM_RANKS = 1 ]; then
  while [ $counter -lt $max_counter ]; do
    echo "Iteration $counter: Running generate_new_checkpoint.py -s 70"
    torchrun --standalone --nproc-per-node 1 generate_new_checkpoint.py -s 70
    echo "Iteration $counter: Running bench_llama_70b.py $@"
    torchrun --standalone --nproc-per-node 1 bench_llama_70b.py "$@"
    ((counter++))
  done
elif [ $NUM_RANKS = 2 ]; then
  while [ $counter -lt $max_counter ]; do
    echo "Iteration $counter: Running generate_new_checkpoint.py -s 70 -n"
    CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc-per-node 2 generate_new_checkpoint.py -s 70 -n
    echo "Iteration $counter: Running bench_llama_70b.py $@ -n"
    CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc-per-node 2 bench_llama_70b.py "$@" -n
    ((counter++))
  done
else
  echo -n "Bad argument for --num_ranks $NUM_RANKS"
fi
