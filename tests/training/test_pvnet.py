from datetime import datetime

from torch.utils.data.datapipes.iter import IterableWrapper

from ocf_datapipes.training.pvnet import (
    construct_sliced_data_pipeline,
    pvnet_datapipe,
)
from ocf_datapipes.batch import BatchKey, NWPBatchKey
from ocf_datapipes.utils import Location


def test_construct_sliced_data_pipeline(pvnet_config_filename):
    # This is randomly chosen, but real, GSP location
    loc_pipe = IterableWrapper([Location(x=246699.328125, y=849771.9375, id=18)])

    # Also randomly chosen to be in the middle of the test data
    t0_pipe = IterableWrapper([datetime(2020, 4, 1, 13, 30)])

    dp = construct_sliced_data_pipeline(
        pvnet_config_filename,
        location_pipe=loc_pipe,
        t0_datapipe=t0_pipe,
    )

    batch = next(iter(dp))
    assert BatchKey.nwp in batch
    for nwp_source in batch[BatchKey.nwp].keys():
        assert NWPBatchKey.nwp in batch[BatchKey.nwp][nwp_source]


def test_pvnet_datapipe(pvnet_config_filename):
    start_time = datetime(1900, 1, 1)
    end_time = datetime(2050, 1, 1)

    dp = pvnet_datapipe(
        pvnet_config_filename,
        start_time=start_time,
        end_time=end_time,
    )

    batch = next(iter(dp))
    assert BatchKey.nwp in batch
    for nwp_source in batch[BatchKey.nwp].keys():
        assert NWPBatchKey.nwp in batch[BatchKey.nwp][nwp_source]
