from datetime import timedelta

from ocf_datapipes.convert import ConvertNWPToNumpyBatch
from ocf_datapipes.select import SelectLiveT0Time
from ocf_datapipes.transform.xarray import AddT0IdxAndSamplePeriodDuration, ConvertToNWPTargetTime
from ocf_datapipes.utils.consts import BatchKey


def test_convert_nwp_to_numpy_batch(nwp_datapipe):
    nwp_datapipe = AddT0IdxAndSamplePeriodDuration(
        nwp_datapipe,
        sample_period_duration=timedelta(minutes=60),
        history_duration=timedelta(minutes=60),
    )
    t0_datapipe = SelectLiveT0Time(nwp_datapipe, dim_name="init_time_utc")

    nwp_datapipe = ConvertToNWPTargetTime(
        nwp_datapipe,
        t0_datapipe=t0_datapipe,
        sample_period_duration=timedelta(minutes=60),
        history_duration=timedelta(minutes=60),
        forecast_duration=timedelta(minutes=60),
    )
    nwp_datapipe = ConvertNWPToNumpyBatch(nwp_datapipe)
    data = next(iter(nwp_datapipe))
    assert BatchKey.nwp in data
    assert BatchKey.nwp_channel_names in data
