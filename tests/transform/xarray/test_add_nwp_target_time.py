from datetime import timedelta

from ocf_datapipes.select import SelectLiveT0Time
from ocf_datapipes.transform.xarray import ConvertToNWPTargetTime


def test_add_nwp_target_time(nwp_datapipe):
    t0_datapipe = SelectLiveT0Time(nwp_datapipe, dim_name="init_time_utc")
    nwp_datapipe = ConvertToNWPTargetTime(
        nwp_datapipe,
        t0_datapipe,
        sample_period_duration=timedelta(minutes=60),
        history_duration=timedelta(hours=2),
        forecast_duration=timedelta(hours=3),
    )
    data = next(iter(nwp_datapipe))
    assert "target_time_utc" in data.coords
    assert len(data.coords["target_time_utc"]) == 6
