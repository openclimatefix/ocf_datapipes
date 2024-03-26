import numpy as np
import torch

from ocf_datapipes.batch import BatchKey, NumpyBatch, TensorBatch
from ocf_datapipes.batch.utils import copy_batch_to_device, batch_to_tensor
import pytest

@pytest.fixture()
def sample_numpy_batch():
    sample: NumpyBatch = {}
    sample[BatchKey.satellite_actual] = np.full((12, 10, 24, 24), 0)
    return sample


def test_batch_to_tensor(sample_numpy_batch):
    tensor_batch = batch_to_tensor(sample_numpy_batch)
    assert isinstance(tensor_batch[BatchKey.satellite_actual], torch.Tensor)


def test_copy_batch_to_device(sample_numpy_batch):
    tensor_batch = batch_to_tensor(sample_numpy_batch)
    device = torch.device("cpu")
    batch_copy: TensorBatch = copy_batch_to_device(tensor_batch, device)
    assert batch_copy[BatchKey.satellite_actual].device == device  # type: ignore
