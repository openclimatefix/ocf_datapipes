"""Additional utils for working with batches"""
import numpy as np
import torch


def copy_batch_to_device(batch: dict, device: torch.device) -> dict:
    """
    Moves a dict-batch of tensors to new device.

    Args:
        batch: dict with tensors to move
        device: Device to move tensors to

    Returns:
        A dict with tensors moved to new device
    """
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


def batch_to_tensor(batch: dict) -> dict:
    """
    Moves numpy batch to a tensor

    Args:
        batch: dict-like batch with data in numpy arrays

    Returns:
        A batch with data in torch tensors
    """
    for k, v in batch.items():
        if isinstance(v, dict):
            # Recursion to reach the nested NWP
            batch[k] = batch_to_tensor(v)
        elif isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
            batch[k] = torch.as_tensor(v)
    return batch
