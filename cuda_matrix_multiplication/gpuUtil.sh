#!/bin/bash
export CUDA_VISIBLE_DEVICES=0  # Use only the first GPU
# Start monitoring GPU utilization in the background
# -i only monitor the first GPU
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -l 1 -i 0 > gpu_usage.txt &

# Save the PID of the background monitoring process
monitor_pid=$!

# Run your command
./speedup 20000 20000 20000

# Kill the background monitoring process
kill $monitor_pid

# Optional: Process the output file as needed
