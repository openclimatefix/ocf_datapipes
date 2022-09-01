from datetime import timedelta

from ocf_datapipes.select import SelectLiveT0Time, SelectLiveTimeSlice
from ocf_datapipes.transform.xarray import ConvertToNWPTargetTime


def test_select_hrv(sat_hrv_dp):
    time_len = len(next(iter(sat_hrv_dp)).time_utc.values)
    t0_dp = SelectLiveT0Time(sat_hrv_dp)
    sat_hrv_dp = SelectLiveTimeSlice(
        sat_hrv_dp,
        t0_datapipe=t0_dp,
        history_duration=timedelta(hours=1),
    )
    data = next(iter(sat_hrv_dp))
    assert len(data.time_utc.values) == 13
    assert len(data.time_utc.values) < time_len


def test_select_gsp(gsp_dp):
    time_len = len(next(iter(gsp_dp)).time_utc.values)
    t0_dp = SelectLiveT0Time(gsp_dp)
    gsp_dp = SelectLiveTimeSlice(
        gsp_dp,
        t0_datapipe=t0_dp,
        history_duration=timedelta(hours=2),
    )
    data = next(iter(gsp_dp))
    assert len(data.time_utc.values) == 5
    assert len(data.time_utc.values) < time_len


def test_select_nwp(nwp_dp):
    t0_dp = SelectLiveT0Time(nwp_dp, dim_name="init_time_utc")
    nwp_dp = ConvertToNWPTargetTime(
        nwp_dp,
        t0_datapipe=t0_dp,
        sample_period_duration=timedelta(hours=1),
        history_duration=timedelta(hours=2),
        forecast_duration=timedelta(hours=3),
    )
    data = next(iter(nwp_dp))
    assert len(data.target_time_utc.values) == 6


def test_select_passiv(passiv_dp):
    time_len = len(next(iter(passiv_dp)).time_utc.values)
    t0_dp = SelectLiveT0Time(passiv_dp)
    passiv_dp = SelectLiveTimeSlice(
        passiv_dp,
        t0_datapipe=t0_dp,
        history_duration=timedelta(hours=1),
    )
    data = next(iter(passiv_dp))
    assert len(data.time_utc.values) == 13
    assert len(data.time_utc.values) < time_len


def test_select_pvoutput(pvoutput_dp):
    time_len = len(next(iter(pvoutput_dp)).time_utc.values)
    t0_dp = SelectLiveT0Time(pvoutput_dp, dim_name="time_utc")
    pvoutput_dp = SelectLiveTimeSlice(
        pvoutput_dp,
        t0_datapipe=t0_dp,
        history_duration=timedelta(hours=1),
    )
    data = next(iter(pvoutput_dp))
    assert len(data.time_utc.values) == 13
    assert len(data.time_utc.values) < time_len
