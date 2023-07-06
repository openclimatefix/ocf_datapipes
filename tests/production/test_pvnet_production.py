from datetime import datetime

from freezegun import freeze_time
from torchdata.datapipes.iter import IterableWrapper

from ocf_datapipes.training.pvnet import (
    construct_sliced_data_pipeline,
)
from ocf_datapipes.utils.consts import Location
import pytest


@freeze_time("2020-04-02 02:30:00")
def test_construct_sliced_data_pipeline_outside_test(configuration_filename, gsp_yields):
    # This is randomly chosen, but real, GSP location
    loc_pipe = IterableWrapper([Location(x=246699.328125, y=849771.9375, id=18)])

    # Chosen to lie beyond end of test data
    t0_pipe = IterableWrapper([datetime(2020, 4, 2, 0, 30)])

    dp = construct_sliced_data_pipeline(
        configuration_filename,
        location_pipe=loc_pipe,
        t0_datapipe=t0_pipe,
        production=True,
    )

    batch = next(iter(dp))


@freeze_time("2020-04-01 02:30:00")
def test_construct_sliced_data_pipeline(configuration_filename, gsp_yields):
    # This is randomly chosen, but real, GSP location
    loc_pipe = IterableWrapper([Location(x=246699.328125, y=849771.9375, id=18)])

    # Also randomly chosen to be in the middle of the test data
    t0_pipe = IterableWrapper([datetime(2020, 4, 1, 13, 30)])

    dp = construct_sliced_data_pipeline(
        configuration_filename,
        location_pipe=loc_pipe,
        t0_datapipe=t0_pipe,
        check_satellite_no_zeros=True,
        production=True,
    )

    batch = next(iter(dp))


def test_construct_sliced_data_pipeline_satellite_with_zeros(configuration_filename, gsp_yields):
    # This is randomly chosen, but real, GSP location
    loc_pipe = IterableWrapper([Location(x=246699.328125, y=849771.9375, id=18)])

    # Data after the test data
    t0_pipe = IterableWrapper([datetime(2020, 4, 1, 12, 30)])

    dp = construct_sliced_data_pipeline(
        configuration_filename,
        location_pipe=loc_pipe,
        t0_datapipe=t0_pipe,
        check_satellite_no_zeros=True,
        production=True,
    )
    with pytest.raises(Exception):
        batch = next(iter(dp))
