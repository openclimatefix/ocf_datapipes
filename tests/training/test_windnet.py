from datetime import datetime
import numpy as np
import xarray as xr

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
    # Need to serialize attributes to strings
    datasets.to_netcdf("test.nc", mode="w", engine="h5netcdf", compute=True)
    dp = windnet_netcdf_datapipe(
        filenames=["test.nc"],
        keys=[
            "nwp",
            "sat",
            "wind",
        ],
    )
    datasets = next(iter(dp))


def test_windnet_datapipe_nwp_channels(configuration_filename):
    start_time = datetime(1900, 1, 1)
    end_time = datetime(2050, 1, 1)
    dp = windnet_datapipe(
        configuration_filename,
        start_time=start_time,
        end_time=end_time,
    )
    datasets = next(iter(dp))

    # expand channels from "t" to ["t", "wind"]
    nwp_ukv = datasets.__getitem__("nwp-ukv")
    s = np.concatenate((nwp_ukv.values, nwp_ukv.values), axis=1)
    da = xr.DataArray(
        data=s,
        dims=nwp_ukv.dims,
        coords={
            "nwp-ukv__channel": ("nwp-ukv__channel", ["t", "wind"]),
            "nwp-ukv__y_osgb": nwp_ukv.coords["nwp-ukv__y_osgb"],
            "nwp-ukv__x_osgb": nwp_ukv.coords["nwp-ukv__x_osgb"],
            "ukv__target_time_utc": nwp_ukv.coords["nwp-ukv__target_time_utc"],
            "nwp-ukv__init_time_utc": nwp_ukv.coords["nwp-ukv__init_time_utc"],
            "nwp-ukv__step": nwp_ukv.coords["nwp-ukv__step"],
        },
        attrs=nwp_ukv.attrs,
    )
    datasets.__setitem__("nwp-ukv", da)

    # Need to serialize attributes to strings
    datasets.to_netcdf("test.nc", mode="w", engine="h5netcdf", compute=True)
    dp = windnet_netcdf_datapipe(
        filenames=["test.nc"],
        keys=[
            "nwp",
            "sat",
            "wind",
        ],
        nwp_channels={"ukv": ["t"]},
    )
    datasets = next(iter(dp))
