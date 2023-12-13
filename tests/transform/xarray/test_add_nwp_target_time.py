from datetime import timedelta

from ocf_datapipes.transform.xarray import ConvertToNWPTargetTime
from torch.utils.data.datapipes.iter import IterableWrapper


def test_convert_to_nwp_target_time(nwp_datapipe):
    t0_datapipe = IterableWrapper([next(iter(nwp_datapipe)).init_time_utc.values[-1]])
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
