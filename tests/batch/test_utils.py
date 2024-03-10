import numpy as np
import torch

from ocf_datapipes.batch import BatchKey, NumpyBatch
from ocf_datapipes.batch import copy_batch_to_device, batch_to_tensor


def _create_test_batch() -> NumpyBatch:
    sample: NumpyBatch = {}
    sample[BatchKey.satellite_actual] = np.full((12, 10, 24, 24), 0)
    return sample


def test_batch_to_tensor():
    batch = _create_test_batch()
    tensor_batch = batch_to_tensor(batch)
    assert isinstance(tensor_batch[BatchKey.satellite_actual], torch.Tensor)


def test_copy_batch_to_device():
    batch = _create_test_batch()
    tensor_batch = batch_to_tensor(batch)
    device = torch.device("cpu")
    batch_copy = copy_batch_to_device(tensor_batch, device)
    assert batch_copy[BatchKey.satellite_actual].device == device
