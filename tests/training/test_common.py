import pytest
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.iter import Zipper
from ocf_datapipes.config.model import Configuration
from ocf_datapipes.utils.consts import Location
from torch.utils.data import DataLoader
from ocf_datapipes.training.common import (
    add_selected_time_slices_from_datapipes,
    get_and_return_overlapping_time_periods_and_t0,
    open_and_return_datapipes,
    create_t0_and_loc_datapipes,
)

import fsspec
from pyaml_env import parse_config

import pandas as pd
import numpy as np


def test_open_and_return_datapipes():
    used_datapipes = open_and_return_datapipes("tests/config/test.yaml")
    for key in ["nwp", "topo", "gsp", "pv", "sat", "hrv"]:
        assert key in used_datapipes.keys()
        assert isinstance(used_datapipes[key], IterDataPipe)
    assert sorted(list(used_datapipes.keys())) == sorted(
        ["nwp", "config", "topo", "gsp", "pv", "sat", "hrv"]
    )


def test_get_and_return_overlapping_time_periods_and_t0():
    used_datapipes = open_and_return_datapipes("tests/config/test.yaml")
    used_datapipes = get_and_return_overlapping_time_periods_and_t0(used_datapipes)
    for key in [
        "gsp",
        "gsp_t0",
        "hrv",
        "hrv_t0",
        "nwp",
        "nwp_t0",
        "pv",
        "pv_t0",
        "sat",
        "sat_t0",
        "topo",
    ]:
        assert key in used_datapipes.keys()
        assert isinstance(used_datapipes[key], IterDataPipe)
    assert sorted(list(used_datapipes.keys())) == sorted(
        [
            "config",
            "gsp",
            "gsp_t0",
            "hrv",
            "hrv_t0",
            "nwp",
            "nwp_t0",
            "pv",
            "pv_t0",
            "sat",
            "sat_t0",
            "topo",
        ]
    )


def test_add_selected_time_slices_from_datapipes():
    used_datapipes = open_and_return_datapipes("tests/config/test.yaml")
    used_datapipes = get_and_return_overlapping_time_periods_and_t0(used_datapipes)
    used_datapipes = add_selected_time_slices_from_datapipes(used_datapipes)
    for key in ["nwp", "topo", "gsp", "gsp_future", "pv", "pv_future", "sat", "hrv"]:
        assert key in used_datapipes.keys()
        assert isinstance(used_datapipes[key], IterDataPipe)
    assert sorted(list(used_datapipes.keys())) == sorted(
        ["nwp", "config", "topo", "gsp", "gsp_future", "pv", "pv_future", "sat", "hrv"]
    )


@pytest.mark.skip("Too long for GitHub CI")
def test_add_selected_time_slices_from_datapipes_fork_iterations():
    used_datapipes = open_and_return_datapipes("tests/config/test.yaml")
    used_datapipes = get_and_return_overlapping_time_periods_and_t0(used_datapipes)
    used_datapipes = add_selected_time_slices_from_datapipes(used_datapipes)
    for key in ["nwp", "topo", "gsp", "gsp_future", "pv", "pv_future", "sat", "hrv"]:
        assert key in used_datapipes.keys()
        assert isinstance(used_datapipes[key], IterDataPipe)
    assert sorted(list(used_datapipes.keys())) == sorted(
        ["nwp", "config", "topo", "gsp", "gsp_future", "pv", "pv_future", "sat", "hrv"]
    )
    # Zip together to see if any are missing
    zipped = Zipper(
        *[
            used_datapipes[k]
            for k in ["nwp", "topo", "gsp", "gsp_future", "pv", "pv_future", "sat", "hrv"]
        ]
    )
    dataloader = DataLoader(zipped)
    for i, batch in enumerate(dataloader):
        _ = batch
        if i + 1 % 50000 == 0:
            break


def test_create_t0_and_loc_datapipes():
    datapipes_dict = open_and_return_datapipes("tests/config/test.yaml")

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
