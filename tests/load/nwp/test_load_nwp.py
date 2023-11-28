import numpy as np

from ocf_datapipes.load import OpenNWP


def test_load_nwp():
    nwp_datapipe = OpenNWP(zarr_path="tests/data/nwp_data/test.zarr")
    metadata = next(iter(nwp_datapipe))
    assert metadata is not None


def test_load_icon_eu():
    nwp_datapipe = OpenNWP(
        zarr_path="tests/data/icon_eu.zarr",
        provider="icon-eu",
    )
    metadata = next(iter(nwp_datapipe))
    assert metadata is not None


def test_load_icon_global():
    nwp_datapipe = OpenNWP(
        zarr_path="tests/data/icon_global.zarr",
        provider="icon-global",
    )
    metadata = next(iter(nwp_datapipe))
    assert metadata is not None


def test_load_ecmwf():
    nwp_datapipe = OpenNWP(
        zarr_path="tests/data/ifs.zarr",
        provider="ecmwf",
    )
    metadata = next(iter(nwp_datapipe))
    assert metadata is not None
