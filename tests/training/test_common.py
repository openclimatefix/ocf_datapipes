from datetime import datetime
import numpy as np
import pytest

from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.iter import Zipper
from torch.utils.data import DataLoader

from ocf_datapipes.config.model import Configuration
from ocf_datapipes.utils import Location
from ocf_datapipes.training.common import (
    add_selected_time_slices_from_datapipes,
    get_and_return_overlapping_time_periods_and_t0,
    open_and_return_datapipes,
    create_t0_and_loc_datapipes,
    construct_loctime_pipelines,
    potentially_coarsen,
)


def test_open_and_return_datapipes(configuration_filename):
    used_datapipes = open_and_return_datapipes(configuration_filename)
    expected_keys = set(["nwp", "config", "wind", "topo", "gsp", "pv", "sat", "hrv"])
    assert set(used_datapipes.keys()) == expected_keys
    for key in expected_keys - set(["nwp", "config"]):
        assert isinstance(used_datapipes[key], IterDataPipe)

    for nwp_key in used_datapipes["nwp"].keys():
        assert isinstance(used_datapipes["nwp"][nwp_key], IterDataPipe)


def test_get_and_return_overlapping_time_periods_and_t0(configuration_filename):
    used_datapipes = open_and_return_datapipes(configuration_filename)
    used_datapipes = get_and_return_overlapping_time_periods_and_t0(used_datapipes)

    datapipe_keys = set(["gsp", "hrv", "nwp/ukv", "pv", "sat", "wind"])
    t0_keys = set([f"{k}_t0" for k in datapipe_keys])
    extra_keys = set(["config", "topo"])

    assert set(used_datapipes.keys()) == datapipe_keys.union(t0_keys, extra_keys)

    for key in datapipe_keys.union(t0_keys):
        assert isinstance(used_datapipes[key], IterDataPipe)


def test_add_selected_time_slices_from_datapipes(configuration_filename):
    used_datapipes = open_and_return_datapipes(configuration_filename)
    used_datapipes = get_and_return_overlapping_time_periods_and_t0(used_datapipes)
    used_datapipes = add_selected_time_slices_from_datapipes(used_datapipes)

    datapipe_keys = set(
        ["gsp", "gsp_future", "pv", "pv_future", "hrv", "nwp/ukv", "sat", "wind", "wind_future"]
    )
    extra_keys = set(["config", "topo"])

    assert set(used_datapipes.keys()) == datapipe_keys.union(extra_keys)

    for key in datapipe_keys:
        assert isinstance(used_datapipes[key], IterDataPipe)

    # Zip datapipes together
    zipped = Zipper(*[used_datapipes[k] for k in datapipe_keys])
    batch = next(iter(zipped))


@pytest.mark.skip("Too long for GitHub CI")
def test_add_selected_time_slices_from_datapipes_fork_iterations(configuration_filename):
    used_datapipes = open_and_return_datapipes(configuration_filename)
    used_datapipes = get_and_return_overlapping_time_periods_and_t0(used_datapipes)
    used_datapipes = add_selected_time_slices_from_datapipes(used_datapipes)

    datapipe_keys = set(
        [
            "gsp",
            "gsp_future",
            "pv",
            "pv_future",
            "hrv",
            "nwp/ukv",
            "sat",
            "topo",
            "wind",
            "wind_future",
        ]
    )

    # Zip datapipes together
    zipped = Zipper(*[used_datapipes[k] for k in datapipe_keys])
    dataloader = DataLoader(zipped)
    for i, batch in zip(range(50_000), dataloader):
        pass


def test_create_t0_and_loc_datapipes(configuration_filename):
    datapipes_dict = open_and_return_datapipes(configuration_filename)

    configuration = datapipes_dict.pop("config")

    del datapipes_dict["pv"]

    location_pipe, t0_datapipe = create_t0_and_loc_datapipes(
        datapipes_dict,
        configuration,
        key_for_t0="gsp",
        shuffle=True,
    )

    assert isinstance(location_pipe, IterDataPipe)
    assert isinstance(t0_datapipe, IterDataPipe)

    loc0, t0 = next(iter(location_pipe.zip(t0_datapipe)))
    assert isinstance(loc0, Location)
    assert isinstance(t0, np.datetime64)


def test_construct_loctime_pipelines(configuration_filename):
    start_time = datetime(1900, 1, 1)
    end_time = datetime(2050, 1, 1)

    loc_pipe, t0_pipe = construct_loctime_pipelines(
        configuration_filename,
        start_time=start_time,
        end_time=end_time,
    )

    next(iter(loc_pipe))
    next(iter(t0_pipe))


def test_potentially_coarsen(nwp_gfs_data):
    """ The nwp_gfs_data has lat and long of 0 to 9"""

    assert nwp_gfs_data.si10.shape[2:] == (10, 10)
    data = potentially_coarsen(xr_data=nwp_gfs_data, coarsen_to_deg=2)
    print(data)
    # should be now 0,2,4,6,8
    assert data.si10.shape[2:] == (5, 5)

    data = potentially_coarsen(xr_data=nwp_gfs_data, coarsen_to_deg=3)
    # should be now 0,3,6,9
    assert data.si10.shape[2:] == (4, 4)

    data = potentially_coarsen(xr_data=nwp_gfs_data, coarsen_to_deg=1)
    # should be the same
    assert data.si10.shape[2:] == (10, 10)

