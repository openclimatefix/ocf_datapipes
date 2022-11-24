from ocf_datapipes.training.common import (
    get_and_return_overlapping_time_periods_and_t0,
    open_and_return_datapipes,
    add_selected_time_slices_from_datapipes,
)
from torchdata.datapipes.iter import IterDataPipe, Zipper
import pytest


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
    for key in ["nwp", "topo", "gsp", "gsp_future", "pv", "sat", "hrv"]:
        assert key in used_datapipes.keys()
        assert isinstance(used_datapipes[key], IterDataPipe)
    assert sorted(list(used_datapipes.keys())) == sorted(
        ["nwp", "config", "topo", "gsp", "gsp_future", "pv", "sat", "hrv"]
    )

from torchdata.dataloader2 import DataLoader2

@pytest.mark.skip("Too long for GitHub CI")
def test_add_selected_time_slices_from_datapipes_fork_iterations():
    used_datapipes = open_and_return_datapipes("tests/config/test.yaml")
    used_datapipes = get_and_return_overlapping_time_periods_and_t0(used_datapipes)
    used_datapipes = add_selected_time_slices_from_datapipes(used_datapipes)
    for key in ["nwp", "topo", "gsp", "gsp_future", "pv", "sat", "hrv"]:
        assert key in used_datapipes.keys()
        assert isinstance(used_datapipes[key], IterDataPipe)
    assert sorted(list(used_datapipes.keys())) == sorted(
        ["nwp", "config", "topo", "gsp", "gsp_future", "pv", "sat", "hrv"]
    )
    # Zip together to see if any are missing
    zipped = Zipper(
        *[used_datapipes[k] for k in ["nwp", "topo", "gsp", "gsp_future", "pv", "sat", "hrv"]]
    )
    dataloader = DataLoader2(zipped)
    for i, batch in enumerate(dataloader):
        _ = batch
        if i + 1 % 50000 == 0:
            break

