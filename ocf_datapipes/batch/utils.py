"""Additional utils for working with batches"""

import numpy as np
import torch

from ocf_datapipes.batch import NumpyBatch, TensorBatch


def _copy_batch_to_device(batch: dict, device: torch.device) -> dict:
    """
    Moves tensor leaves in a nested dict to a new device

    Args:
        batch: nested dict with tensors to move
        device: Device to move tensors to

    Returns:
        A dict with tensors moved to new device
    """
    batch_copy = {}

    for k, v in batch.items():
        if isinstance(v, dict):
            # Recursion to reach the nested NWP
            batch_copy[k] = _copy_batch_to_device(v, device)
        elif isinstance(v, torch.Tensor):
            batch_copy[k] = v.to(device)
        else:
            batch_copy[k] = v
    return batch_copy


def copy_batch_to_device(batch: TensorBatch, device: torch.device) -> TensorBatch:
    """
    Moves the tensors in a TensorBatch to a new device.

    Args:
        batch: TensorBatch with tensors to move
        device: Device to move tensors to

    Returns:
        TensorBatch with tensors moved to new device
    """
    return _copy_batch_to_device(batch, device)


def _batch_to_tensor(batch: dict) -> dict:
    """
    Moves ndarrays in a nested dict to torch tensors

    Args:
        batch: nested dict with data in numpy arrays

    Returns:
        Nested dict with data in torch tensors
    """
    for k, v in batch.items():
        if isinstance(v, dict):
            # Recursion to reach the nested NWP
            batch[k] = _batch_to_tensor(v)
        elif isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
            batch[k] = torch.as_tensor(v)
    return batch


def batch_to_tensor(batch: NumpyBatch) -> TensorBatch:
    """
    Moves data in a NumpyBatch to a TensorBatch

    Args:
        batch: NumpyBatch with data in numpy arrays

    Returns:
        TensorBatch with data in torch tensors
    """
    return _batch_to_tensor(batch)
