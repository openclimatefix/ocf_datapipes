from datetime import datetime

from freezegun import freeze_time
from torchdata.datapipes.iter import IterableWrapper

from ocf_datapipes.training.pvnet import (
    construct_sliced_data_pipeline,
)
from ocf_datapipes.utils.consts import Location


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
