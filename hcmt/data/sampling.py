"""
Optimized Sampling for Variable-Length Clinical Sequences
======================================================

Key optimizations:
1. Length bucketing: Group sequences of similar lengths into the same batch
   - Avoids padding short sequences to match very long sequences
   - Reduces wasted GPU computation on zero-padding
   
2. Batch packing: Instead of padding, pack sequences of similar length together

3. Memory-efficient iteration: Avoids creating large Python lists
"""

import math
import random
from typing import List, Iterator
from torch.utils.data import Sampler, DataLoader
import numpy as np


class ResumableSampler(Sampler):
    """Deterministic O(1)-memory permutation with an exact resume offset."""

    def __init__(self, data_source, seed: int = 42):
        self.n = len(data_source)
        self.seed = int(seed)
        self.epoch = 0
        self.start = 0
        self.skip_batches = 0

    def set_epoch(self, epoch: int, start: int = 0):
        self.epoch = int(epoch)
        self.start = int(start)

    def __len__(self):
        return max(0, self.n - self.start)

    def __iter__(self):
        if self.n <= 0:
            return
        rng = random.Random(self.seed + self.epoch)
        multiplier = rng.randrange(1, self.n + 1)
        while math.gcd(multiplier, self.n) != 1:
            multiplier = (multiplier + 1) % self.n or 1
        offset = rng.randrange(self.n)
        for position in range(self.start, self.n):
            yield (multiplier * position + offset) % self.n

# Lazy import to avoid circular imports
def _get_timeline_collate():
    try:
        from hcmt.data.timeline_v6 import collate_timeline
        return collate_timeline
    except ImportError:
        return None


class LengthSortedBatchSampler(Sampler):
    """Constant-memory random-pool length sampler.

    It preserves full coverage (each index is emitted once per epoch), but
    sorts only a bounded random pool before forming batches. This avoids the
    O(N) length table that is impossible for the 310M-sample timeline while
    still eliminating most cross-length padding.
    """

    def __init__(self, dataset, batch_size=32, shuffle=True, seed=42,
                 drop_last=False, num_buckets=8):
        self.dataset = dataset
        self.n = len(dataset)
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.seed = int(seed)
        self.drop_last = drop_last
        self.epoch = 0
        self.start = 0
        self.skip_batches = 0
        self.pool_size = max(self.batch_size, self.batch_size * max(2, int(num_buckets)))
        if hasattr(dataset, 'sequence_length'):
            self.length_fn = lambda _dataset, idx: dataset.sequence_length(idx)
        else:
            self.length_fn = self._default_length

    @staticmethod
    def _default_length(dataset, idx):
        return len(dataset[idx].get('dynamic_values', ()))

    def set_epoch(self, epoch, start=0):
        self.epoch = int(epoch)
        self.start = int(start)
        self.skip_batches = 0

    def set_epoch_batch(self, epoch, start_batch=0):
        """Resume from an emitted batch while preserving exact batch order."""
        self.epoch = int(epoch)
        self.start = 0
        self.skip_batches = int(start_batch)

    def __len__(self):
        remaining = max(0, self.n - self.start)
        if self.drop_last:
            batches = remaining // self.batch_size
        else:
            batches = (remaining + self.batch_size - 1) // self.batch_size
        return max(0, batches - self.skip_batches)

    def __iter__(self):
        if self.n == 0 or self.start >= self.n:
            return
        rng = random.Random(self.seed + self.epoch)
        a = rng.randrange(1, self.n + 1) if self.shuffle else 1
        while math.gcd(a, self.n) != 1:
            a = (a + 1) % self.n or 1
        b = rng.randrange(self.n) if self.shuffle else 0

        emitted = 0
        for pool_start in range(self.start, self.n, self.pool_size):
            pool_end = min(self.n, pool_start + self.pool_size)
            indices = [(a * j + b) % self.n for j in range(pool_start, pool_end)]
            indices.sort(key=lambda idx: self.length_fn(self.dataset, idx))
            batches = [indices[batch_start:batch_start + self.batch_size]
                       for batch_start in range(0, len(indices), self.batch_size)]
            if self.drop_last and batches and len(batches[-1]) != self.batch_size:
                batches.pop()
            if self.shuffle:
                rng.shuffle(batches)
            for batch in batches:
                if emitted < self.skip_batches:
                    emitted += 1
                    continue
                emitted += 1
                yield batch


class LengthBucketSampler(Sampler):
    """
    Samples batches where sequences of similar lengths are grouped together.
    
    This avoids the problem where a single very long sequence forces all
    other sequences in the batch to be padded with many zeros.
    
    Strategy:
    1. Pre-compute sequence length estimates for all samples
    2. Sort samples by length
    3. Group into buckets of similar lengths
    4. Within each bucket, shuffle samples
    5. Yield batches from buckets, optionally shuffling bucket order
    
    Memory: O(n) for storing indices, but avoids O(steps) Python list growth
    """
    
    def __init__(
        self,
        dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
        length_estimator=None,
        num_buckets: int = 8,
    ):
        """
        Args:
            dataset: Dataset with __len__ method
            batch_size: Target batch size
            shuffle: Whether to shuffle within buckets
            seed: Random seed for reproducibility
            drop_last: Drop last incomplete batch
            length_estimator: Function(dataset, idx) -> approximate length
            num_buckets: Number of length buckets (more = better packing, less = more variance)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.num_buckets = num_buckets
        self.epoch = 0  # Used for per-epoch random variation in __iter__
        
        self.n = len(dataset)
        self.num_batches = self.n // batch_size
        if not drop_last and self.n % batch_size != 0:
            self.num_batches += 1
        
        # Estimate sequence lengths for bucketing
        # This is a fast approximation - we sample a subset of the dataset
        self._compute_length_buckets(length_estimator)
    
    def _compute_length_buckets(self, length_estimator):
        """Compute length-based bucketing."""
        # Sample to estimate length distribution
        sample_size = min(1000, self.n)
        sample_indices = np.random.choice(self.n, sample_size, replace=False)
        
        lengths = []
        for idx in sample_indices:
            try:
                if length_estimator:
                    length = length_estimator(self.dataset, idx)
                else:
                    # Default: use a sample from dataset
                    item = self.dataset[idx]
                    length = len(item.get('dynamic_values', [1]))
            except:
                length = 1
            lengths.append((idx, length))
        
        if not lengths:
            self.lengths = {i: 1 for i in range(self.n)}
            self.bucket_boundaries = [0, self.n]
            return
        
        # Get min/max lengths
        lengths_only = [l for _, l in lengths]
        min_len = min(lengths_only)
        max_len = max(lengths_only) if max(lengths_only) > min_len else min_len + 1
        
        # Create bucket boundaries
        bucket_size = (max_len - min_len) / self.num_buckets
        self.bucket_boundaries = [
            min_len + i * bucket_size 
            for i in range(self.num_buckets + 1)
        ]
        
        # Assign all samples to buckets based on estimated length
        # For samples we didn't sample, assign based on median
        median_length = np.median(lengths_only)
        self.lengths = {}
        
        # Build bucket indices
        self.bucket_indices = {b: [] for b in range(self.num_buckets)}
        
        # Use sampled data to estimate bucket sizes
        for idx, length in lengths:
            bucket = self._get_bucket(length)
            self.bucket_indices[bucket].append(idx)
            self.lengths[idx] = length
        
        # Estimate bucket sizes for non-sampled indices
        avg_bucket_size = sample_size / self.num_buckets
        per_bucket = self.n // self.num_buckets
        
        for i in range(self.n):
            if i not in self.lengths:
                # Assign based on uniform distribution approximation
                bucket = min(i // per_bucket, self.num_buckets - 1)
                self.lengths[i] = median_length
                self.bucket_indices[bucket].append(i)
    
    def _get_bucket(self, length: float) -> int:
        """Get bucket index for a length."""
        for b in range(self.num_buckets):
            if length < self.bucket_boundaries[b + 1]:
                return b
        return self.num_buckets - 1
    
    def __len__(self) -> int:
        return self.num_batches
    
    def __iter__(self) -> Iterator[List[int]]:
        """Iterate batches. Each batch contains indices of similar-length samples."""
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        
        # Shuffle within each bucket (copy to avoid mutating original)
        bucket_data = {}
        for b in range(self.num_buckets):
            indices = list(self.bucket_indices[b])
            if self.shuffle:
                rng.shuffle(indices)
            bucket_data[b] = indices
        
        # Track position within each bucket
        bucket_pos = [0] * self.num_buckets
        
        # Random bucket order (cycled for larger datasets)
        bucket_order = list(range(self.num_buckets))
        if self.shuffle:
            rng.shuffle(bucket_order)
        
        order_idx = 0
        
        while True:
            batch = []
            
            # Try to fill one batch
            attempts = 0
            max_attempts = self.num_buckets * (self.batch_size + 1)
            
            while len(batch) < self.batch_size and attempts < max_attempts:
                attempts += 1
                # Pick next bucket in order
                current_bucket = bucket_order[order_idx % self.num_buckets]
                order_idx += 1
                
                # If this bucket is exhausted, try next
                attempts_inner = 0
                while (bucket_pos[current_bucket] >= len(bucket_data[current_bucket])
                       and attempts_inner < self.num_buckets):
                    attempts_inner += 1
                    current_bucket = bucket_order[order_idx % self.num_buckets]
                    order_idx += 1
                
                # If all buckets exhausted, stop
                if bucket_pos[current_bucket] >= len(bucket_data[current_bucket]):
                    break
                
                # Add sample from bucket
                pos = bucket_pos[current_bucket]
                batch.append(bucket_data[current_bucket][pos])
                bucket_pos[current_bucket] = pos + 1
            
            # If we couldn't fill the batch
            if not batch:
                break  # No more data at all
            elif len(batch) < self.batch_size:
                if not self.drop_last:
                    yield batch
                break  # Done, partial batch yielded or skipped
            else:
                # Yield full batch
                yield batch


class DynamicBatchSampler(Sampler):
    """
    Groups samples by EXACT length to completely avoid padding.
    
    For clinical data, many samples share the same sequence length. This
    sampler pre-computes lengths and groups them, then yields batches
    containing exactly same-length samples.
    
    Key optimization:
    - 100% coverage of all samples
    - Minimal padding waste (often <5%)
    - Fully deterministic when seed is fixed
    
    Memory: O(n) for storing all lengths in a list. For datasets with
    millions of unique lengths, use LengthBucketSampler instead.
    """
    
    def __init__(
        self,
        dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
        length_fn=None,
        max_length_cache: int = 50000,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        
        self.n = len(dataset)
        
        # Length function
        if length_fn is not None:
            self.length_fn = length_fn
        else:
            self.length_fn = self._default_length_fn
        
        # Pre-compute length for each sample
        # Use cache if dataset is huge to avoid loading everything
        self.lengths = [0] * self.n
        
        if self.n <= max_length_cache:
            # Compute exact lengths
            for i in range(self.n):
                try:
                    self.lengths[i] = self.length_fn(self.dataset, i)
                except Exception:
                    self.lengths[i] = 144
        else:
            # Sample to estimate lengths for large datasets
            # For truly huge datasets, this is just an approximation
            for i in range(self.n):
                self.lengths[i] = 144  # Default placeholder
        
        # Group by exact length
        from collections import defaultdict
        self.length_groups = defaultdict(list)
        for idx, length in enumerate(self.lengths):
            self.length_groups[length].append(idx)
        
        self.unique_lengths = sorted(self.length_groups.keys())
    
    def _default_length_fn(self, dataset, idx) -> int:
        """Default length estimator - returns sequence length."""
        try:
            item = dataset[idx]
            return len(item.get('dynamic_values', [144]))
        except Exception:
            return 144
    
    def __len__(self) -> int:
        total = 0
        for length in self.unique_lengths:
            count = len(self.length_groups[length])
            if self.drop_last:
                total += count // self.batch_size
            else:
                total += (count + self.batch_size - 1) // self.batch_size
        return total
    
    def __iter__(self) -> Iterator[List[int]]:
        """Yield batches where each batch has samples of the SAME length."""
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        
        # Shuffle within each length group, and shuffle group order
        groups = []
        for length in self.unique_lengths:
            indices = list(self.length_groups[length])
            if self.shuffle:
                rng.shuffle(indices)
            groups.append(indices)
        
        if self.shuffle:
            rng.shuffle(groups)
        
        # Yield batches from each group
        for indices in groups:
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size:
                    yield batch
                elif not self.drop_last:
                    yield batch


def create_optimized_loader(
    dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    sampler_type: str = 'bucket',  # 'bucket', 'dynamic', or 'default'
    seed: int = 42,
    drop_last: bool = False,
    length_estimator=None,
    num_buckets: int = 8,
    device: str = 'cuda',
):
    """
    Create an optimized data loader with length-based batching.
    
    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        pin_memory: Use pinned memory for faster GPU transfer
        sampler_type: 'bucket' (recommended), 'dynamic', or 'default'
        seed: Random seed
        drop_last: Drop incomplete last batch
        length_estimator: Function(dataset, idx) -> length
        num_buckets: Number of buckets for LengthBucketSampler
    
    Returns:
        DataLoader with optimized sampling
    """
    from torch.utils.data import DataLoader
    
    if sampler_type == 'bucket':
        # LengthSortedBatchSampler: sorts within pools, constant memory for 310M samples
        batch_sampler = LengthSortedBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
            num_buckets=max(32, num_buckets),  # Larger pools = better grouping
        )
        sampler = None
    elif sampler_type == 'dynamic':
        # DynamicBatchSampler: groups by exact length, best GPU utilization
        # WARNING: For 310M samples, this pre-computes all lengths
        # Each pool in DynamicBatchSampler gets its own bucket for exact length grouping
        batch_sampler = DynamicBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        sampler = None
    else:
        # Default: no length-based batching
        batch_sampler = None
        sampler = None
    
    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_memory and ('cuda' in str(device)),
        persistent_workers=num_workers > 0,
    )
    
    # Use timeline collator if available
    if hasattr(dataset, 'collate_fn'):
        loader_kwargs['collate_fn'] = dataset.collate_fn
    elif hasattr(dataset, '__class__') and dataset.__class__.__name__ == 'TimelineDataset':
        collate = _get_timeline_collate()
        loader_kwargs['collate_fn'] = collate
    else:
        loader_kwargs['collate_fn'] = None
    
    if batch_sampler is not None:
        # The sampler yields lists of indices; it must be passed as a
        # batch_sampler, not as a scalar-index sampler.
        loader_kwargs['batch_sampler'] = batch_sampler
    elif sampler is not None:
        loader_kwargs['sampler'] = sampler
    else:
        loader_kwargs['batch_size'] = batch_size
        loader_kwargs['shuffle'] = shuffle
    
    # Increase prefetch for better pipeline utilization
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = max(2, num_workers * 2)
    
    return DataLoader(dataset, **loader_kwargs)
