from datetime import datetime

from ocf_datapipes.training.windnet import (
    windnet_datapipe,
    windnet_netcdf_datapipe,
)
from ocf_datapipes.utils.utils import combine_to_single_dataset, uncombine_from_single_dataset
import pytest


def test_windnet_datapipe(configuration_filename):
    start_time = datetime(1900, 1, 1)
    end_time = datetime(2050, 1, 1)
    dp = windnet_datapipe(
        configuration_filename,
        start_time=start_time,
        end_time=end_time,
    )
    datasets = next(iter(dp))
    dataset = combine_to_single_dataset(datasets)
    # Need to serialize attributes to strings
    dataset.to_netcdf("test.nc", mode="w", engine="h5netcdf", compute=True)
    dp = windnet_netcdf_datapipe(
        config_filename=configuration_filename,
        filenames=["test.nc"],
        keys=["gsp", "nwp", "sat", "pv"],
    )
    datasets = next(iter(dp))
