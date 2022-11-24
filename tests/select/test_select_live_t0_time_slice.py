from datetime import timedelta

import pytest

from ocf_datapipes.select import SelectLiveT0Time, SelectLiveTimeSlice
from ocf_datapipes.transform.xarray import ConvertToNWPTargetTime


@pytest.mark.skip()
def test_select_hrv(sat_hrv_datapipe):
    time_len = len(next(iter(sat_hrv_datapipe)).time_utc.values)
    t0_datapipe = SelectLiveT0Time(sat_hrv_datapipe)
    sat_hrv_datapipe = SelectLiveTimeSlice(
        sat_hrv_datapipe,
        t0_datapipe=t0_datapipe,
        history_duration=timedelta(hours=1),
    )
    data = next(iter(sat_hrv_datapipe))
    assert len(data.time_utc.values) == 13
    assert len(data.time_utc.values) < time_len


def test_select_gsp(gsp_datapipe):
    time_len = len(next(iter(gsp_datapipe)).time_utc.values)
    t0_datapipe = SelectLiveT0Time(gsp_datapipe)
    gsp_datapipe = SelectLiveTimeSlice(
        gsp_datapipe,
        t0_datapipe=t0_datapipe,
        history_duration=timedelta(hours=2),
    )
    data = next(iter(gsp_datapipe))
    assert len(data.time_utc.values) == 5
    assert len(data.time_utc.values) < time_len


@pytest.mark.skip("Too long")
def test_select_nwp(nwp_datapipe):
    t0_datapipe = SelectLiveT0Time(nwp_datapipe, dim_name="init_time_utc")
    nwp_datapipe = ConvertToNWPTargetTime(
        nwp_datapipe,
        t0_datapipe=t0_datapipe,
        sample_period_duration=timedelta(hours=1),
        history_duration=timedelta(hours=2),
        forecast_duration=timedelta(hours=3),
    )
    data = next(iter(nwp_datapipe))
    assert len(data.target_time_utc.values) == 6


def test_select_passiv(passiv_datapipe):
    time_len = len(next(iter(passiv_datapipe)).time_utc.values)
    t0_datapipe = SelectLiveT0Time(passiv_datapipe)
    passiv_datapipe = SelectLiveTimeSlice(
        passiv_datapipe,
        t0_datapipe=t0_datapipe,
        history_duration=timedelta(hours=1),
    )
    data = next(iter(passiv_datapipe))
    assert len(data.time_utc.values) == 13
    assert len(data.time_utc.values) < time_len


def test_select_pvoutput(pvoutput_datapipe):
    time_len = len(next(iter(pvoutput_datapipe)).time_utc.values)
    t0_datapipe = SelectLiveT0Time(pvoutput_datapipe, dim_name="time_utc")
    pvoutput_datapipe = SelectLiveTimeSlice(
        pvoutput_datapipe,
        t0_datapipe=t0_datapipe,
        history_duration=timedelta(hours=1),
    )
    data = next(iter(pvoutput_datapipe))
    assert len(data.time_utc.values) == 13
    assert len(data.time_utc.values) < time_len
