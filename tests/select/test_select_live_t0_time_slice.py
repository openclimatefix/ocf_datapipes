from ocf_datapipes.select import SelectLiveT0TimeSlice
from datetime import timedelta

def test_select_hrv(sat_hrv_dp):
    time_len = len(next(iter(sat_hrv_dp)).time_utc.values)
    sat_hrv_dp = SelectLiveT0TimeSlice(sat_hrv_dp, history_duration=timedelta(minutes=60))
    data = next(iter(sat_hrv_dp))
    assert len(data.time_utc.values) == 13
    assert len(data.time_utc.values) < time_len

def test_select_gsp(gsp_dp):
    time_len = len(next(iter(gsp_dp)).time_utc.values)
    gsp_dp = SelectLiveT0TimeSlice(gsp_dp, history_duration=timedelta(minutes=120))
    data = next(iter(gsp_dp))
    assert len(data.time_utc.values) == 5
    assert len(data.time_utc.values) < time_len

def test_select_nwp(nwp_dp):
    time_len = len(next(iter(nwp_dp)).time_utc.values)
    nwp_dp = SelectLiveT0TimeSlice(nwp_dp, history_duration=timedelta(minutes=120))
    data = next(iter(nwp_dp))
    assert len(data.time_utc.values) == 3
    assert len(data.time_utc.values) < time_len

def test_select_passiv(passiv_dp):
    time_len = len(next(iter(passiv_dp)).time_utc.values)
    passiv_dp = SelectLiveT0TimeSlice(passiv_dp, history_duration=timedelta(minutes=60))
    data = next(iter(passiv_dp))
    assert len(data.time_utc.values) == 13
    assert len(data.time_utc.values) < time_len

def test_select_pvoutput(pvoutput_dp):
    time_len = len(next(iter(pvoutput_dp)).time_utc.values)
    pvoutput_dp = SelectLiveT0TimeSlice(pvoutput_dp, history_duration=timedelta(minutes=60))
    data = next(iter(pvoutput_dp))
    assert len(data.time_utc.values) == 13
    assert len(data.time_utc.values) < time_len
