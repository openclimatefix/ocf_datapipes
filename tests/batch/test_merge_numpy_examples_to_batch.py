import pytest
import numpy as np
from torch.utils.data.datapipes.iter import IterableWrapper

from ocf_datapipes.batch import BatchKey, NWPBatchKey, NumpyBatch, NWPNumpyBatch


from ocf_datapipes.batch.merge_numpy_examples_to_batch import (
    unstack_np_batch_into_examples,
    MergeNumpyBatchIterDataPipe,
    MergeNumpyExamplesToBatchIterDataPipe,
)


def _single_batch_sample(fill_value):
    """This function allows us to create batches with different filled values"""
    sample: NumpyBatch = {}
    sample[BatchKey.satellite_actual] = np.full(
        (12, 10, 24, 24), fill_value
    )  # shape: (time, channel, x, y)
    sample[BatchKey.gsp_id] = np.full((1,), fill_value)  # shape: (1,)
    sample[BatchKey.gsp_t0_idx] = 4  # scalar and constant across all samples

    sample_nwp_ukv: NWPNumpyBatch = {}
    sample_nwp_ukv[NWPBatchKey.nwp] = np.full(
        (8, 2, 24, 24), fill_value
    )  # shape: (time, variable, x, y)
    sample_nwp_ukv[NWPBatchKey.nwp_channel_names] = ["a", "b"]  # shape: (variable,)

    sample_nwp_ecmwf: NWPNumpyBatch = {}
    sample_nwp_ecmwf[NWPBatchKey.nwp] = np.full(
        (8, 4, 12, 12), fill_value
    )  # shape: (time, variable, x, y)

    sample[BatchKey.nwp] = {
        "ukv": sample_nwp_ukv,
        "ecmwf": sample_nwp_ecmwf,
    }

    return sample

def _single_batch_sample_nan(fill_value):
    """This function allows us to create batches with different filled values"""
    
    sample: NumpyBatch = {}
    sample[BatchKey.satellite_actual] = np.full(
        (12, 10, 24, 24), fill_value, dtype=np.float32
    ) # shape: (time, channel, x, y)
    sample[BatchKey.gsp_id] = np.full((1,), fill_value)  # shape: (1,)
    sample[BatchKey.gsp_t0_idx] = 4  # scalar and constant across all samples

    sample_nwp_ukv: NWPNumpyBatch = {}
    sample_nwp_ukv[NWPBatchKey.nwp] = np.full(
        (8, 2, 24, 24), fill_value, dtype=np.float32
    )# shape: (time, variable, x, y)
    sample_nwp_ukv[NWPBatchKey.nwp][0,0,0,0] = np.nan

    sample_nwp_ukv[NWPBatchKey.nwp_channel_names] = ["a", "b"]  # shape: (variable,)

    sample_nwp_ecmwf: NWPNumpyBatch = {}
    sample_nwp_ecmwf[NWPBatchKey.nwp] = np.full(
        (8, 4, 12, 12), fill_value
    )  # shape: (time, variable, x, y)

    sample[BatchKey.nwp] = {
        "ukv": sample_nwp_ukv,
        "ecmwf": sample_nwp_ecmwf,
    }
    # print(sample[BatchKey.nwp]["ukv"])
    return sample


@pytest.fixture
def numpy_sample_datapipe():
    dp = IterableWrapper([_single_batch_sample(i) for i in range(8)])
    return dp

@pytest.fixture
def numpy_nan_sample_datapipe():
    dp = IterableWrapper([_single_batch_sample_nan(i) for i in range(8)])
    return dp

def test_merge_numpy_batch(numpy_sample_datapipe):
    dp = MergeNumpyBatchIterDataPipe(numpy_sample_datapipe.batch(4))
    dp_iter = iter(dp)

    for i in range(2):
        batch = next(dp_iter)
        assert (
            batch[BatchKey.satellite_actual][:, 0, 0, 0, 0] == np.arange(4 * i, 4 * (i + 1))
        ).all()
        assert batch[BatchKey.gsp_t0_idx] == 4

        nwp_batch = batch[BatchKey.nwp]["ukv"]
        assert (nwp_batch[NWPBatchKey.nwp][:, 0, 0, 0, 0] == np.arange(4 * i, 4 * (i + 1))).all()
        assert nwp_batch[NWPBatchKey.nwp_channel_names] == ["a", "b"]

def test_merge_numpy_batch_for_nans(numpy_nan_sample_datapipe):
    with pytest.raises(ValueError):  # checks for Error raised if NWP/BatchKey DataArray contains Nans
        dp = MergeNumpyBatchIterDataPipe(numpy_nan_sample_datapipe.batch(4))
        dp_iter = iter(dp)
        metadata = next(dp_iter)

def test_merge_numpy_examples_to_batch(numpy_sample_datapipe):
    dp = MergeNumpyExamplesToBatchIterDataPipe(numpy_sample_datapipe, n_examples_per_batch=4)
    dp_iter = iter(dp)

    for i in range(2):
        batch = next(dp_iter)
        assert (
            batch[BatchKey.satellite_actual][:, 0, 0, 0, 0] == np.arange(4 * i, 4 * (i + 1))
        ).all()
        assert batch[BatchKey.gsp_t0_idx] == 4

        nwp_batch = batch[BatchKey.nwp]["ukv"]
        assert (nwp_batch[NWPBatchKey.nwp][:, 0, 0, 0, 0] == np.arange(4 * i, 4 * (i + 1))).all()
        assert nwp_batch[NWPBatchKey.nwp_channel_names] == ["a", "b"]


def test_unstack_np_batch_into_examples(numpy_sample_datapipe):
    dp = MergeNumpyBatchIterDataPipe(numpy_sample_datapipe.batch(4))

    batch = next(iter(dp))
    samples = unstack_np_batch_into_examples(batch)

    for i, sample in enumerate(samples):
        assert sample[BatchKey.satellite_actual][0, 0, 0, 0] == i
        assert sample[BatchKey.gsp_t0_idx] == 4

        nwp_sample = sample[BatchKey.nwp]["ukv"]
        assert nwp_sample[NWPBatchKey.nwp][0, 0, 0, 0] == i
        assert nwp_sample[NWPBatchKey.nwp_channel_names] == ["a", "b"]
