# config.py
# Shared constants across train.py, validate.py, hyperparameter_search.py

import os

# Worker allocation: reserve 2 cores for OS, cap at 16 for HPC
MAX_WORKERS = 16

def get_num_workers(is_cuda: bool) -> int:
    """Return the appropriate number of dataloader workers for the current hardware."""
    if is_cuda:
        return min(MAX_WORKERS, max(1, os.cpu_count() - 2))
    return 0  # MPS and CPU both need 0