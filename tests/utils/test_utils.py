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
