from ocf_datapipes.select import SelectLiveTimeSlice, SelectLiveT0Time
from ocf_datapipes.transform.xarray import AddNWPTargetTime
from datetime import timedelta
from freezegun import freeze_time

def test_select_hrv(sat_hrv_dp):
    time_len = len(next(iter(sat_hrv_dp)).time_utc.values)
    sat_hrv_dp = SelectLiveTimeSlice(sat_hrv_dp, history_duration=timedelta(minutes=60))
    data = next(iter(sat_hrv_dp))
    assert len(data.time_utc.values) == 13
    assert len(data.time_utc.values) < time_len

def test_select_gsp(gsp_dp):
    time_len = len(next(iter(gsp_dp)).time_utc.values)
    gsp_dp = SelectLiveTimeSlice(gsp_dp, history_duration=timedelta(minutes=120))
    data = next(iter(gsp_dp))
    assert len(data.time_utc.values) == 5
    assert len(data.time_utc.values) < time_len

@freeze_time("2020-04-01 12:00:00")  # Date in middle of NWP time
def test_select_nwp(nwp_dp):
    t0_dp = SelectLiveT0Time(nwp_dp, dim_name="init_time_utc")
    nwp_dp = AddNWPTargetTime(nwp_dp, t0_dp, sample_period_duration=timedelta(hours=1), history_duration=timedelta(hours=2), forecast_duration=timedelta(hours=8))
    data = next(iter(nwp_dp))
    print(data)
    print(data.init_time_utc.values)
    time_len = len(next(iter(nwp_dp)).target_time_utc.values)
    nwp_dp = SelectLiveTimeSlice(nwp_dp, t0_datapipe=t0_dp, history_duration=timedelta(minutes=120), dim_name="target_time_utc")
    data = next(iter(nwp_dp))
    assert len(data.time_utc.values) == 3
    assert len(data.time_utc.values) < time_len

def test_select_passiv(passiv_dp):
    time_len = len(next(iter(passiv_dp)).time_utc.values)
    passiv_dp = SelectLiveTimeSlice(passiv_dp, history_duration=timedelta(minutes=60))
    data = next(iter(passiv_dp))
    assert len(data.time_utc.values) == 13
    assert len(data.time_utc.values) < time_len

def test_select_pvoutput(pvoutput_dp):
    time_len = len(next(iter(pvoutput_dp)).time_utc.values)
    pvoutput_dp = SelectLiveTimeSlice(pvoutput_dp, history_duration=timedelta(minutes=60))
    data = next(iter(pvoutput_dp))
    assert len(data.time_utc.values) == 13
    assert len(data.time_utc.values) < time_len
