import numpy as np
from ocf_datapipes.utils.utils import searchsorted
from ocf_datapipes.utils.utils import combine_to_single_dataset, uncombine_from_single_dataset
from ocf_datapipes.training.windnet import windnet_datapipe
from datetime import datetime
import xarray as xr


def test_searchsorted():
    ys = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    assert searchsorted(ys, 2.1) == 2
    ys_r = np.array([5, 4, 3, 2, 1], dtype=np.float32)
    assert searchsorted(ys_r, 2.1, assume_ascending=False) == 3


def test_combine_uncombine_from_single_dataset(configuration_filename):
    start_time = datetime(1900, 1, 1)
    end_time = datetime(2050, 1, 1)
    dp = windnet_datapipe(
        configuration_filename,
        start_time=start_time,
        end_time=end_time,
    )
    datasets = next(iter(dp))
    dataset: xr.Dataset = combine_to_single_dataset(datasets)
    assert isinstance(dataset, xr.Dataset)
    multiple_datasets = uncombine_from_single_dataset(dataset)
    for key in multiple_datasets.keys():
        for i in range(len(multiple_datasets[key].time_utc)):
            assert datasets[key][i].equals(multiple_datasets[key].isel(time_utc=i))
