"""Additional utils for working with batches"""
import numpy as np
import torch


def copy_batch_to_device(batch, device):
    """Moves a dict-batch of tensors to new device."""
    batch_copy = {}

    for k, v in batch.items():
        if isinstance(v, dict):
            # Recursion to reach the nested NWP
            batch_copy[k] = copy_batch_to_device(v, device)
        elif isinstance(v, torch.Tensor):
            batch_copy[k] = v.to(device)
        else:
            batch_copy[k] = v
    return batch_copy


def batch_to_tensor(batch):
    """Moves numpy batch to a tensor"""
    for k, v in batch.items():
        if isinstance(v, dict):
            # Recursion to reach the nested NWP
            batch[k] = batch_to_tensor(v)
        elif isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
            batch[k] = torch.as_tensor(v)
    return batch
