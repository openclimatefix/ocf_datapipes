from ocf_datapipes.convert import ConvertNWPToNumpyBatch
from ocf_datapipes.transform.xarray import AddT0IdxAndSamplePeriodDuration, ConvertToNWPTargetTime
from ocf_datapipes.select import SelectLiveT0Time

from datetime import timedelta


def test_convert_nwp_to_numpy_batch(nwp_dp):
    nwp_dp = AddT0IdxAndSamplePeriodDuration(
        nwp_dp, sample_period_duration=timedelta(minutes=60), history_duration=timedelta(minutes=60)
    )
    t0_dp = SelectLiveT0Time(nwp_dp, dim_name="init_time_utc")

    nwp_dp = ConvertToNWPTargetTime(
        nwp_dp,
        t0_datapipe=t0_dp,
        sample_period_duration=timedelta(minutes=60),
        history_duration=timedelta(minutes=60),
        forecast_duration=timedelta(minutes=60),
    )
    nwp_dp = ConvertNWPToNumpyBatch(nwp_dp)
    data = next(iter(nwp_dp))
    assert data is not None
