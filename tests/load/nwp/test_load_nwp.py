import numpy as np

from ocf_datapipes.load import OpenNWP
from ocf_datapipes.load.nwp.nwp import OpenLatestNWPDataPipe


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


def test_load_latest_nwp():
    base_nwp_datapipe = OpenNWP(zarr_path="tests/data/nwp_data/test.zarr")
    recent_obs_datapipe = OpenLatestNWPDataPipe(base_nwp_datapipe)
    data = next(iter(recent_obs_datapipe))
    assert isinstance(
        data.init_time_utc.values, np.datetime64
    )  # single time observation, rather than array
