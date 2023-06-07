from datetime import timedelta
import pandas as pd
import numpy as np
from torchdata.datapipes.iter import IterableWrapper
from ocf_datapipes.select import SelectTimeSlice


def test_select_time_slice_sat(sat_datapipe):
    data = next(iter(sat_datapipe))

    t0_datapipe = IterableWrapper(pd.to_datetime(data.time_utc.values)[3:6])

    # Check with history and forecast durations
    dp = SelectTimeSlice(
        sat_datapipe,
        t0_datapipe,
        sample_period_duration=timedelta(minutes=5),
        history_duration=timedelta(minutes=5),
        forecast_duration=timedelta(minutes=5),
    )

    sat_samples = list(dp)
    t0_values = list(t0_datapipe)

    for sat_sample, t0 in zip(sat_samples, t0_values):
        assert len(sat_sample.time_utc) == 3
        assert sat_sample.time_utc[1] == t0

    # Check again with intervals
    dp = SelectTimeSlice(
        sat_datapipe,
        t0_datapipe,
        sample_period_duration=timedelta(minutes=5),
        interval_start=timedelta(minutes=-5),
        interval_end=timedelta(minutes=5),
    )

    sat_samples = list(dp)

    for sat_sample, t0 in zip(sat_samples, t0_values):
        assert len(sat_sample.time_utc) == 3
        assert sat_sample.time_utc[1] == t0

    # Check with out of bounds selection
    t_last = pd.to_datetime(data.time_utc.values[-1])
    t0_values = [
        t_last - timedelta(minutes=5),
        t_last,
        t_last + timedelta(minutes=5),
        t_last + timedelta(minutes=10),
    ]
    t0_datapipe = IterableWrapper(t0_values)

    dp = SelectTimeSlice(
        sat_datapipe,
        t0_datapipe,
        sample_period_duration=timedelta(minutes=5),
        interval_start=timedelta(minutes=-5),
        interval_end=timedelta(minutes=5),
        fill_selection=True,
    )

    sat_samples = list(dp)

    for i, (sat_sample, t0) in enumerate(zip(sat_samples, t0_values)):
        assert len(sat_sample.time_utc) == 3
        assert sat_sample.time_utc[1] == t0
        # Correct number of time steps are all NaN
        sat_sel = sat_sample.isel(x_geostationary=0, y_geostationary=0, channel=0)
        assert np.isnan(sat_sel.values).sum() == i
