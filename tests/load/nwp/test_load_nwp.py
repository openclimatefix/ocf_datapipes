import pandas as pd
from xarray import DataArray

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

def test_load_gfs():
    nwp_datapipe = OpenNWP(
        zarr_path="tests/data/gfs_test.zarr.zip",
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
