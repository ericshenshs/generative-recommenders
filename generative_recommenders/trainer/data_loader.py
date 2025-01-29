# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

import os
from typing import Optional, Tuple

import gin
import torch


@gin.configurable  # Allows this function to be configured via gin
def create_data_loader(
    dataset: torch.utils.data.Dataset,  # Input dataset to load
    batch_size: int,                    # Number of samples per batch
    world_size: int,                    # Total number of distributed processes
    rank: int,                          # Current process rank in distributed setup
    shuffle: bool,                      # Whether to shuffle the dataset
    prefetch_factor: int = 128,         # Number of samples loaded in advance by each worker
    num_workers: Optional[int] = os.cpu_count(),  # Number of subprocesses for data loading
    drop_last: bool = False,            # Whether to drop the last incomplete batch
) -> Tuple[
    Optional[torch.utils.data.distributed.DistributedSampler[torch.utils.data.Dataset]],
    torch.utils.data.DataLoader,
]:
    """Creates a DataLoader with optional distributed sampling support.

    This function sets up a PyTorch DataLoader with distributed training capabilities.
    When shuffle is True, it uses DistributedSampler to handle data partitioning
    across multiple processes in distributed training.

    Args:
        dataset: The dataset to load
        batch_size: How many samples per batch to load
        world_size: Number of processes participating in distributed training
        rank: Process rank within distributed training
        shuffle: If True, data is shuffled and DistributedSampler is used
        prefetch_factor: Number of batches to prefetch per worker
        num_workers: Number of worker processes for data loading
        drop_last: If True, drop the last incomplete batch

    Returns:
        A tuple containing:
        - The DistributedSampler if shuffle is True, None otherwise
        - The configured DataLoader instance
          
    References:
        https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
            - Main PyTorch data loading utility that combines a dataset and sampler
            - Provides single/multi-process data loading with customizable loading order
            - Supports automatic batching, memory pinning, and custom collation
        https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
            - Sampler that restricts data loading to a subset of the dataset in distributed training
            - Automatically partitions dataset across processes/GPUs for parallel training
            - Ensures each process sees different data when shuffling is enabled
    """
    if shuffle:
        # Create a distributed sampler when shuffling is requested
        # This ensures each process gets a different slice of the dataset
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,  # Total number of processes
            rank=rank,                # Current process ID
            shuffle=True,             # Shuffle data within each process
            seed=0,                   # Fixed seed for reproducibility
            drop_last=drop_last,      # Whether to drop last incomplete batch
        )
        # A sampler is needed for distributed training to:
        # 1. Partition data across processes - ensures each GPU/process works on different data
        # 2. Maintain balanced workload - each process gets roughly equal amount of samples
        # 3. Avoid data duplication - prevents multiple processes from processing same samples
        # 4. Enable proper shuffling - coordinates shuffling across processes to prevent overlap
        #
        # The sampler achieves these goals by:
        # a. Using num_replicas to divide dataset size by total processes
        # b. Using rank to determine which subset this process handles
        # c. Maintaining internal indices to track data partitioning
        # d. Coordinating shuffling across processes with a shared seed
        # e. Optionally dropping remainder samples to keep batches balanced
else:
        # No sampler needed when not shuffling because:
        # 1. Data order is deterministic - each process can use simple indexing
        # 2. Each process can calculate its subset using rank and world_size directly
        # 3. No need to coordinate random shuffling across processes
        # 4. Basic sequential access is sufficient for non-shuffled distributed training
        sampler = None

    # Create and return the DataLoader
    #
    # The DataLoader is responsible for:
    # 1. Batching individual samples from the dataset into mini-batches (groups of samples
    #    processed together to balance computational efficiency and memory usage)
    # 2. Shuffling the data if specified (via sampler in distributed mode)
    # 3. Loading data in parallel using multiple worker processes
    # 4. Prefetching batches to optimize GPU utilization
    # 5. Handling the data pipeline from dataset to training loop efficiently
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        # shuffle=True cannot be used with sampler
        num_workers=num_workers or 0,  # Fall back to 0 if num_workers is None
        sampler=sampler,
        # Number of batches to prefetch per worker process
        # Controls how many batches each worker loads ahead of time to reduce I/O bottlenecks
        # Higher values increase memory usage but can improve throughput by overlapping data loading with training
        prefetch_factor=prefetch_factor,
    )
    return sampler, data_loader
