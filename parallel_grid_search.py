#!/usr/bin/env python3
"""
Parallel grid search that runs multiple hyperparameter experiments concurrently
across 8 GPUs. Each GPU runs a different learning rate experiment.
"""
import subprocess
import time
import sys

def main():
    num_gpus = 8

    # Run the faster_grid_search.py with parallel GPU utilization
    # We'll launch it with torchrun to use all GPUs
    cmd = [
        "torchrun",
        "--nproc_per_node=8",
        "faster_grid_search_distributed.py"
    ]

    print("="*80)
    print("LAUNCHING PARALLEL GRID SEARCH ACROSS 8 GPUs")
    print("="*80)
    print(f"Command: {' '.join(cmd)}")
    print("="*80)

    subprocess.run(cmd)

if __name__ == '__main__':
    main()
