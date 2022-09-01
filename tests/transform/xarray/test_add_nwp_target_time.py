from datetime import timedelta

from ocf_datapipes.select import SelectLiveT0Time
from ocf_datapipes.transform.xarray import ConvertToNWPTargetTime


def test_add_nwp_target_time(nwp_dp):
    t0_dp = SelectLiveT0Time(nwp_dp, dim_name="init_time_utc")
    nwp_dp = ConvertToNWPTargetTime(
        nwp_dp,
        t0_dp,
        sample_period_duration=timedelta(minutes=60),
        history_duration=timedelta(hours=2),
        forecast_duration=timedelta(hours=3),
    )
    data = next(iter(nwp_dp))
    assert "target_time_utc" in data.coords
    assert len(data.coords["target_time_utc"]) == 6
