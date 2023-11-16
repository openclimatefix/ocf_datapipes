from datetime import timedelta
from torch.utils.data.datapipes.iter import IterableWrapper
import pandas as pd

from ocf_datapipes.transform.xarray import ConvertToNWPTargetTimeWithDropout


def test_convert_to_nwp_target_time_with_dropout(nwp_datapipe):
    data = next(iter(nwp_datapipe))

    t0 = pd.Timestamp(data.init_time_utc.values[3])
    times = [
        t0 - timedelta(minutes=60),
        t0,
        t0 + timedelta(minutes=30),
        t0 + timedelta(minutes=60),
        t0 + timedelta(minutes=120),
    ]

    t0_datapipe = IterableWrapper(times)

    dropout_datapipe = ConvertToNWPTargetTimeWithDropout(
        nwp_datapipe,
        t0_datapipe,
        sample_period_duration=timedelta(minutes=60),
        history_duration=timedelta(minutes=60),
        forecast_duration=timedelta(minutes=60),
        dropout_timedeltas=[timedelta(minutes=-5)],
        dropout_frac=1,
    )

    # The init-times should be the last 3-hour mutliple before 5-minutes before each time t
    for t, ds in zip(times, dropout_datapipe):
        assert (
            ds.init_time_utc.values == (t - timedelta(minutes=5)).floor(timedelta(hours=3))
        ).all()

    times = [
        t0,
        t0 + timedelta(minutes=30),
        t0 + timedelta(minutes=120),
    ]

    t0_datapipe = IterableWrapper(times)

    # Dropout with one hour delay
    dropout_datapipe = ConvertToNWPTargetTimeWithDropout(
        nwp_datapipe,
        t0_datapipe,
        sample_period_duration=timedelta(minutes=60),
        history_duration=timedelta(minutes=60),
        forecast_duration=timedelta(minutes=0),
        dropout_timedeltas=[timedelta(minutes=-55)],
        dropout_frac=1,
    )

    # The init times should be the last 3-hour mutliple before an hour before each time t
    for t, ds in zip(times, dropout_datapipe):
        assert (
            ds.init_time_utc.values == (t - timedelta(minutes=60)).floor(timedelta(hours=3))
        ).all()
