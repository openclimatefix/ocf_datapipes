from datetime import timedelta
import pandas as pd
from torchdata.datapipes.iter import IterableWrapper
from ocf_datapipes.select import SelectTimeSlice


def test_select_time_slice_sat(sat_datapipe):
    data = next(iter(sat_datapipe))
    
    t0_datapipe = IterableWrapper(
        pd.to_datetime(data.time_utc.values)[3:6]
    )
    
    # Check with history and forecast durations
    dp = SelectTimeSlice(
        sat_datapipe,
        t0_datapipe,
        sample_period_duration=timedelta(minutes=5),
        history_duration = timedelta(minutes=5),
        forecast_duration = timedelta(minutes=5),
    )
    
    sat_samples = list(dp)
    t0_values = list(t0_datapipe)
    
    for sat_sample, t0 in zip(sat_samples, t0_values):
        assert len(sat_sample.time_utc)==3
        assert sat_sample.time_utc[1]==t0
    
    # Check again with intervals
    dp = SelectTimeSlice(
        sat_datapipe,
        t0_datapipe,
        sample_period_duration=timedelta(minutes=5),
        interval_start = timedelta(minutes=-5),
        interval_end = timedelta(minutes=5),
    )
    
    
    sat_samples = list(dp)
    
    for sat_sample, t0 in zip(sat_samples, t0_values):
        assert len(sat_sample.time_utc)==3
        assert sat_sample.time_utc[1]==t0
