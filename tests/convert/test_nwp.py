from datetime import timedelta

from ocf_datapipes.convert import ConvertNWPToNumpyBatch
from torch.utils.data.datapipes.iter import IterableWrapper
from ocf_datapipes.transform.xarray import AddT0IdxAndSamplePeriodDuration
from ocf_datapipes.select import ConvertToNWPTargetTimeWithDropout
from ocf_datapipes.utils.consts import NWPBatchKey


def test_convert_nwp_to_numpy_batch(nwp_datapipe):
    nwp_datapipe = AddT0IdxAndSamplePeriodDuration(
        nwp_datapipe,
        sample_period_duration=timedelta(minutes=60),
        history_duration=timedelta(minutes=60),
    )

    t0_datapipe = IterableWrapper([next(iter(nwp_datapipe)).init_time_utc.values[-1]])

    nwp_datapipe = ConvertToNWPTargetTimeWithDropout(
        nwp_datapipe,
        t0_datapipe=t0_datapipe,
        sample_period_duration=timedelta(minutes=60),
        history_duration=timedelta(minutes=60),
        forecast_duration=timedelta(minutes=60),
    )
    nwp_datapipe = ConvertNWPToNumpyBatch(nwp_datapipe)
    data = next(iter(nwp_datapipe))
    assert NWPBatchKey.nwp in data
    assert NWPBatchKey.nwp_channel_names in data
