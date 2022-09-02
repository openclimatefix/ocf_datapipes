from datetime import timedelta

from ocf_datapipes.transform.numpy import ExtendTimestepsToFuture
from ocf_datapipes.utils.consts import BatchKey


def test_extend_hrv_timesteps_to_future(sat_hrv_np_datapipe):
    before_len = len(next(iter(sat_hrv_np_datapipe))[BatchKey.hrvsatellite_time_utc])
    sat_hrv_datapipe = ExtendTimestepsToFuture(
        sat_hrv_np_datapipe,
        forecast_duration=timedelta(hours=1),
        sample_period_duration=timedelta(minutes=5),
    )
    data = next(iter(sat_hrv_datapipe))
    assert len(data[BatchKey.hrvsatellite_time_utc]) == before_len + 12


def test_extend_gsp_timesteps_to_future(gsp_np_datapipe):
    before_len = len(next(iter(gsp_np_datapipe))[BatchKey.gsp_time_utc])
    gsp_np_datapipe = ExtendTimestepsToFuture(
        gsp_np_datapipe,
        forecast_duration=timedelta(hours=8),
        sample_period_duration=timedelta(minutes=30),
    )
    data = next(iter(gsp_np_datapipe))
    assert len(data[BatchKey.gsp_time_utc]) == before_len + 16


def test_extend_passiv_timesteps_to_future(passiv_np_datapipe):
    before_len = len(next(iter(passiv_np_datapipe))[BatchKey.pv_time_utc])
    passiv_np_datapipe = ExtendTimestepsToFuture(
        passiv_np_datapipe,
        forecast_duration=timedelta(hours=2),
        sample_period_duration=timedelta(minutes=5),
    )
    data = next(iter(passiv_np_datapipe))
    assert len(data[BatchKey.pv_time_utc]) == before_len + 24
