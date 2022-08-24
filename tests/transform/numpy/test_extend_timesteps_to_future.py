from ocf_datapipes.transform.numpy import ExtendTimestepsToFuture
from ocf_datapipes.utils.consts import BatchKey
from datetime import timedelta

def test_extend_hrv_timesteps_to_future(sat_hrv_np_dp):
    before_len = len(next(iter(sat_hrv_np_dp))[BatchKey.hrvsatellite_time_utc])
    sat_hrv_dp = ExtendTimestepsToFuture(sat_hrv_np_dp, forecast_duration=timedelta(hours=1), sample_period_duration=timedelta(minutes=5))
    data = next(iter(sat_hrv_dp))
    assert len(data[BatchKey.hrvsatellite_time_utc]) == before_len + 12
