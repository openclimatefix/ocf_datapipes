from datetime import timedelta
import pandas as pd
import numpy as np
from torchdata.datapipes.iter import IterableWrapper
from ocf_datapipes.select import SelectTimeSlice
from ocf_datapipes.select.select_time_slice import fill_1d_bool_gaps


def test_fill_1d_bool_gaps():
    x = np.array([0, 1, 0, 0, 1, 0, 1, 0])
    y = fill_1d_bool_gaps(x, max_gap=2, fill_ends=False).astype(int)
    assert (np.array([0, 1, 1, 1, 1, 1, 1, 0]) == y).all()

    x = np.array([0, 1, 0, 0, 1, 0, 1, 0])
    y = fill_1d_bool_gaps(x, max_gap=1, fill_ends=True).astype(int)
    assert (np.array([1, 1, 0, 0, 1, 1, 1, 1]) == y).all()
    

def test_select_time_slice_sat(sat_datapipe):
    data = next(iter(sat_datapipe))

    t0_datapipe = IterableWrapper(pd.to_datetime(data.time_utc.values)[3:6])

    # ----------- Check with history and forecast durations -----------
    
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
    
    # ------------------ Check again with intervals -------------------
    
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

    # -------------- Check with out of bounds selection ---------------
    
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
        
    # ------------------- Check with interpolation --------------------
    
    data_times = pd.to_datetime(data.time_utc.values)
    t0_datapipe = IterableWrapper(data_times[[0,1,4,5]])
    
    missing_sat_data = data.sel(time_utc=data_times[[0,2,3,6]])
    missing_sat_datapipe = IterableWrapper([missing_sat_data]).repeat(len(t0_datapipe))
    
    # For each sample the timestamps should be missing in this order
    expected_missing_steps = np.array([
        [True, False, False],
        [False, False, False],
        [False, True, True],
        [True, True, False],
    ])


    dp = SelectTimeSlice(
        missing_sat_datapipe,
        t0_datapipe,
        sample_period_duration=timedelta(minutes=5),
        interval_start=timedelta(minutes=-5),
        interval_end=timedelta(minutes=5),
        fill_selection=True,
        max_steps_gap=1,
    )

    sat_samples = list(dp)
    t0_values = list(t0_datapipe)

    for i in range(len(sat_samples)):
        assert len(sat_samples[i].time_utc) == 3
        assert sat_samples[i].time_utc[1] == t0_values[i]
        # Correct number of time steps are all NaN
        sat_sel = sat_samples[i].isel(x_geostationary=0, y_geostationary=0, channel=0)
        assert (np.isnan(sat_sel.values) == expected_missing_steps[i]).all()
