from datetime import datetime

from torchdata.datapipes.iter import IterableWrapper

from ocf_datapipes.training.pvnet import (
    construct_loctime_pipelines,
    construct_sliced_data_pipeline,
    pvnet_datapipe,
)
from ocf_datapipes.utils.consts import Location
import pytest


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


# TODO currently not stable tests
def test_construct_sliced_data_pipeline(configuration_filename):
    # This is randomly chosen, but real, GSP location
    loc_pipe = IterableWrapper([Location(x=246699.328125, y=849771.9375, id=18)])

    # Also randomly chosen to be in the middle of the test data
    t0_pipe = IterableWrapper([datetime(2020, 4, 1, 13, 30)])

    dp = construct_sliced_data_pipeline(
        configuration_filename,
        location_pipe=loc_pipe,
        t0_datapipe=t0_pipe,
        check_satellite_no_zeros=True,
    )

    batch = next(iter(dp))


# TODO currently not stable tests
def test_construct_sliced_data_pipeline_satellite_with_zeros(configuration_filename):
    # This is randomly chosen, but real, GSP location
    loc_pipe = IterableWrapper([Location(x=246699.328125, y=849771.9375, id=18)])

    # Data after the test data
    t0_pipe = IterableWrapper([datetime(2020, 4, 1, 12, 30)])

    dp = construct_sliced_data_pipeline(
        configuration_filename,
        location_pipe=loc_pipe,
        t0_datapipe=t0_pipe,
        check_satellite_no_zeros=True,
    )
    with pytest.raises(Exception):
        batch = next(iter(dp))


def test_pvnet_datapipe(configuration_filename):
    start_time = datetime(1900, 1, 1)
    end_time = datetime(2050, 1, 1)

    dp = pvnet_datapipe(
        configuration_filename,
        start_time=start_time,
        end_time=end_time,
    )

    batch = next(iter(dp))
