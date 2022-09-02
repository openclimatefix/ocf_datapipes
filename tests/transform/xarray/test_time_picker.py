from datetime import timedelta

from ocf_datapipes.select import SelectOverlappingTimeSlice, SelectTimePeriods, SelectTimeSlice
from ocf_datapipes.transform.xarray import GetContiguousT0TimePeriods


def test_time_picker(sat_hrv_datapipe, passiv_datapipe, gsp_datapipe):
    sat_time_datapipe = GetContiguousT0TimePeriods(
        sat_hrv_datapipe,
        sample_period_duration=timedelta(minutes=5),
        history_duration=timedelta(minutes=60),
        forecast_duration=timedelta(minutes=60),
    )
    pv_time_datapipe = GetContiguousT0TimePeriods(
        passiv_datapipe,
        sample_period_duration=timedelta(minutes=5),
        history_duration=timedelta(minutes=60),
        forecast_duration=timedelta(minutes=60),
    )
    gsp_time_datapipe = GetContiguousT0TimePeriods(
        gsp_datapipe,
        sample_period_duration=timedelta(minutes=5),
        history_duration=timedelta(minutes=60),
        forecast_duration=timedelta(minutes=60),
    )
