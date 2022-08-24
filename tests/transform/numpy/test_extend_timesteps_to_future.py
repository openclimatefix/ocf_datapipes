from ocf_datapipes.transform.numpy import ExtendTimestepsToFuture
from ocf_datapipes.utils.consts import BatchKey
from datetime import timedelta

def test_extend_hrv_timesteps_to_future(sat_hrv_np_dp):
    before_len = len(next(iter(sat_hrv_np_dp))[BatchKey.hrvsatellite_time_utc])
    sat_hrv_dp = ExtendTimestepsToFuture(sat_hrv_np_dp, forecast_duration=timedelta(hours=1), sample_period_duration=timedelta(minutes=5))
    data = next(iter(sat_hrv_dp))
    assert len(data[BatchKey.hrvsatellite_time_utc]) == before_len + 12

def test_extend_gsp_timesteps_to_future(gsp_np_dp):
    before_len = len(next(iter(gsp_np_dp))[BatchKey.gsp_time_utc])
    gsp_np_dp = ExtendTimestepsToFuture(gsp_np_dp, forecast_duration=timedelta(hours=8), sample_period_duration=timedelta(minutes=30))
    data = next(iter(gsp_np_dp))
    assert len(data[BatchKey.gsp_time_utc]) == before_len + 16

def test_extend_passiv_timesteps_to_future(passiv_np_dp):
    before_len = len(next(iter(passiv_np_dp))[BatchKey.pv_time_utc])
    passiv_np_dp = ExtendTimestepsToFuture(passiv_np_dp, forecast_duration=timedelta(hours=2), sample_period_duration=timedelta(minutes=5))
    data = next(iter(passiv_np_dp))
    assert len(data[BatchKey.pv_time_utc]) == before_len + 24
