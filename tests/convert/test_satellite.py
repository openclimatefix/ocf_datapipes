from ocf_datapipes.convert import ConvertSatelliteToNumpyBatch
from ocf_datapipes.transform.xarray import AddT0IdxAndSamplePeriodDuration

from datetime import timedelta


def test_convert_satellite_to_numpy_batch(sat_dp):

    sat_dp = AddT0IdxAndSamplePeriodDuration(
        sat_dp, sample_period_duration=timedelta(minutes=5), history_duration=timedelta(minutes=60)
    )
    sat_dp = ConvertSatelliteToNumpyBatch(sat_dp, is_hrv=False)
    data = next(iter(sat_dp))
    assert data is not None


def test_convert_hrvsatellite_to_numpy_batch(sat_dp):
    sat_dp = AddT0IdxAndSamplePeriodDuration(
        sat_dp, sample_period_duration=timedelta(minutes=5), history_duration=timedelta(minutes=60)
    )
    sat_dp = ConvertSatelliteToNumpyBatch(sat_dp, is_hrv=True)
    data = next(iter(sat_dp))
    assert data is not None
