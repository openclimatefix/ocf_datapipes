import numpy as np
from ocf_datapipes.utils.utils import searchsorted
from ocf_datapipes.utils.utils import combine_to_single_dataset, uncombine_from_single_dataset
import xarray as xr
import pytest


@pytest.fixture()
def xarray_dict_sample(pv_xarray_data, nwp_gfs_data):
    # Use already created data pieces to make batch
    da_pv = pv_xarray_data.rename(dict(datetime="time_utc"))
    da_nwp = nwp_gfs_data.rename(dict(time="init_time_utc")).to_array()

    xarray_sample = {
        "pv": da_pv.isel(time_utc=slice(0, 2)),
        "nwp": da_nwp.isel(init_time_utc=slice(1, 2)),
    }
    return xarray_sample


def test_searchsorted():
    ys = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    assert searchsorted(ys, 2.1) == 2
    ys_r = np.array([5, 4, 3, 2, 1], dtype=np.float32)
    assert searchsorted(ys_r, 2.1, assume_ascending=False) == 3


def test_combine_to_single_dataset(xarray_dict_sample):
    ds_comb = combine_to_single_dataset(xarray_dict_sample)
    # Expected data type
    assert isinstance(ds_comb, xr.Dataset)
    # Expected data variables
    assert set(ds_comb.keys()) == set(xarray_dict_sample.keys())


def test_combine_to_single_dataset_same_init_time(xarray_dict_sample):

    # set nwp init_time_utc to be the same
    xarray_dict_sample["nwp"] = xarray_dict_sample["nwp"].isel(init_time_utc=0)

    # get a vector of init times
    init_time_utcs = [xarray_dict_sample["nwp"].init_time_utc.values] * 11

    # need to have the dims
    # - target_time_utc
    # - channel
    # - latitude
    # - longitude
    # with coordinates of step and init_time_utc
    # rename step to target_time_utc, and add init_time_utc and step
    xarray_dict_sample["nwp"] = xarray_dict_sample["nwp"].rename({"variable": "channel"})
    xarray_dict_sample["nwp"] = xarray_dict_sample["nwp"].rename({"step": "target_time_utc"})
    xarray_dict_sample["nwp"] = xarray_dict_sample["nwp"].assign_coords(
        {"init_time_utc": ("target_time_utc", init_time_utcs)}
    )
    xarray_dict_sample["nwp"] = xarray_dict_sample["nwp"].assign_coords(
        {"step": ("target_time_utc", range(len(init_time_utcs)))}
    )
    # order by target_time_utc, channel, latitude, longitude
    xarray_dict_sample["nwp"] = xarray_dict_sample["nwp"].transpose(
        "target_time_utc", "channel", "latitude", "longitude"
    )

    ds_comb = combine_to_single_dataset(xarray_dict_sample)

    # Expected data type
    assert isinstance(ds_comb, xr.Dataset)
    assert len(ds_comb["nwp__init_time_utc"].values) == 11
    # Expected data variables
    assert set(ds_comb.keys()) == set(xarray_dict_sample.keys())


def test_uncombine_from_single_dataset(xarray_dict_sample):
    ds_comb = combine_to_single_dataset(xarray_dict_sample)

    recompiled_xarray_dict_sample = uncombine_from_single_dataset(ds_comb)

    # Right type
    assert isinstance(recompiled_xarray_dict_sample, type(xarray_dict_sample))

    # Keys are the same
    assert set(recompiled_xarray_dict_sample.keys()) == set(xarray_dict_sample.keys())

    # xarray object under each key is the same
    for k in xarray_dict_sample.keys():
        # Right type
        assert isinstance(recompiled_xarray_dict_sample[k], type(xarray_dict_sample[k]))

        # Coord values are the same
        for coord in xarray_dict_sample[k].coords:
            coord_values_same = (
                recompiled_xarray_dict_sample[k][coord].values
                == xarray_dict_sample[k][coord].values
            ).all()
            assert coord_values_same, f"Coord values under key: {coord} are different"

        # Values are the same
        np.testing.assert_array_equal(
            recompiled_xarray_dict_sample[k].values, xarray_dict_sample[k].values
        )
