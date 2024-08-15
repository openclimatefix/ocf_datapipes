import zarr
import shutil
import numpy as np
import pandas as pd
from xarray import DataArray
import pytest
from ocf_datapipes.load import OpenNWP


def test_load_nwp():
    nwp_datapipe = OpenNWP(zarr_path="tests/data/nwp_data/test.zarr")
    metadata = next(iter(nwp_datapipe))
    assert metadata is not None
    dim_keys = set(["channel", "init_time_utc", "y_osgb", "x_osgb", "step"])
    if bool(dim_keys - set(metadata.dims)):
        raise ValueError(
            "The following dimensions are missing: %s" % (str(dim_keys - set(metadata.dims)))
        )


def test_load_icon_eu():
    nwp_datapipe = OpenNWP(
        zarr_path="tests/data/icon_eu.zarr",
        provider="icon-eu",
    )
    metadata = next(iter(nwp_datapipe))
    assert metadata is not None
    dim_keys = set(["isobaricInhPa", "init_time_utc", "latitude", "longitude", "step"])
    if bool(dim_keys - set(metadata.dims)):
        raise ValueError(
            "The following dimensions are missing: %s" % (str(dim_keys - set(metadata.dims)))
        )


def test_load_icon_global():
    nwp_datapipe = OpenNWP(
        zarr_path="tests/data/icon_global.zarr",
        provider="icon-global",
    )
    metadata = next(iter(nwp_datapipe))
    assert metadata is not None
    dim_keys = set(["isobaricInhPa", "init_time_utc", "step"])
    if bool(dim_keys - set(metadata.dims)):
        raise ValueError(
            "The following dimensions are missing: %s" % (str(dim_keys - set(metadata.dims)))
        )


def test_load_ecmwf():
    nwp_datapipe = OpenNWP(
        zarr_path="tests/data/ifs.zarr",
        provider="ecmwf",
    )
    metadata = next(iter(nwp_datapipe))
    assert metadata is not None
    assert type(next(enumerate(metadata))[1]) == DataArray
    dim_keys = set(["channel", "init_time_utc", "latitude", "longitude", "step"])
    if bool(dim_keys - set(metadata.dims)):
        raise ValueError(
            "The following dimensions are missing: %s" % (str(dim_keys - set(metadata.dims)))
        )


def test_load_merra2():
    nwp_datapipe = OpenNWP(
        zarr_path="tests/data/merra2_test.zarr",
        provider="merra2",
    )
    metadata = next(iter(nwp_datapipe))
    assert metadata is not None
    assert type(next(enumerate(metadata))[1]) == DataArray
    dim_keys = set(["channel", "init_time_utc", "latitude", "longitude", "step"])
    if bool(dim_keys - set(metadata.dims)):
        raise ValueError(
            "The following dimensions are missing: %s" % (str(dim_keys - set(metadata.dims)))
        )


def test_load_excarta():
    zarrs = []
    for issue_date in pd.date_range(start="2023-01-01", periods=7, freq="D"):
        zarrs.append(
            issue_date.strftime(
                "https://storage.googleapis.com/excarta-public-us/hindcast/20220225/%Y/%Y%m%d.zarr"
            )
        )

    nwp_datapipe = OpenNWP(
        zarr_path=zarrs,
        provider="excarta",
    )
    metadata = next(iter(nwp_datapipe))
    assert metadata is not None


def test_load_excarta_local():
    nwp_datapipe = OpenNWP(
        zarr_path="tests/data/excarta/hindcast.zarr",
        provider="excarta",
    )
    metadata = next(iter(nwp_datapipe))
    assert metadata is not None
    dim_keys = set(["channel", "init_time_utc", "latitude", "longitude", "step"])
    if bool(dim_keys - set(metadata.dims)):
        raise ValueError(
            "The following dimensions are missing: %s" % (str(dim_keys - set(metadata.dims)))
        )


def test_check_for_zeros():
    # to generate data with zeros and limits:
    original_store_path = "tests/data/nwp_data/test.zarr"
    original_store = zarr.open(original_store_path, mode="r")
    new_store_path = "tests/data/nwp_data/test_with_zeros_n_limits_n_nans.zarr"
    # Optionally, clear the destination store if it already exists
    shutil.rmtree(new_store_path, ignore_errors=True)
    with zarr.open(new_store_path, mode="w") as new_store:
        for item in original_store:
            zarr.copy(original_store[item], new_store, name=item)

        new_store["UKV"][0, 0, 0, 0] = 0
        new_store["UKV"][0, 0, 0, 1] = np.random.uniform(190, 360, size=(548,))
        new_store["UKV"][0, 0, 0, 2] = np.nan

    shutil.copy(
        "tests/data/nwp_data/test.zarr/.zmetadata",
        "tests/data/nwp_data/test_with_zeros_n_limits_n_nans.zarr/.zmetadata",
    )

    # positive test case
    nwp_datapipe1 = OpenNWP(
        zarr_path=new_store_path,
        check_for_zeros=True,
    )
    with pytest.raises(ValueError):  # checks for Error raised if NWP DataArray contains zeros
        metadata = next(iter(nwp_datapipe1))

    # negative test case
    nwp_datapipe2 = OpenNWP(zarr_path=original_store_path, check_for_zeros=True)
    metadata = next(iter(nwp_datapipe2))
    assert metadata is not None


def test_check_physical_limits():
    # positive test case
    nwp_datapipe1 = OpenNWP(
        zarr_path="tests/data/nwp_data/test_with_zeros_n_limits_n_nans.zarr", check_physical_limits=True
    )
    with pytest.raises(
        ValueError
    ):  # checks for Error raised if NWP data UKV is outside physical limits
        metadata = next(iter(nwp_datapipe1))

    # negative test case
    nwp_datapipe2 = OpenNWP(zarr_path="tests/data/nwp_data/test.zarr", check_physical_limits=True)
    metadata = next(iter(nwp_datapipe2))
    assert metadata is not None

def test_check_if_nans():
    # positive test case
    nwp_datapipe1 = OpenNWP(
        zarr_path="tests/data/nwp_data/test_with_zeros_n_limits_n_nans.zarr", check_for_nans=True
    )
    with pytest.raises(ValueError):  # checks for Error raised if NWP DataArray contains nans
        metadata = next(iter(nwp_datapipe1))
    
    # negative test case
    nwp_datapipe2 = OpenNWP(zarr_path="tests/data/nwp_data/test.zarr", check_for_nans=True)
    metadata = next(iter(nwp_datapipe2))
    assert metadata is not None

    shutil.rmtree(
        "tests/data/nwp_data/test_with_zeros_n_limits_n_nans.zarr"
    )  # removes the zarr file created for testing