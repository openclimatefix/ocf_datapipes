from datetime import timedelta

from ocf_datapipes.convert import ConvertSatelliteToNumpyBatch
from ocf_datapipes.transform.xarray import AddT0IdxAndSamplePeriodDuration
from ocf_datapipes.utils.consts import BatchKey


def test_convert_satellite_to_numpy_batch(sat_datapipe):
    sat_datapipe = AddT0IdxAndSamplePeriodDuration(
        sat_datapipe,
        sample_period_duration=timedelta(minutes=5),
        history_duration=timedelta(minutes=60),
    )
    sat_datapipe = ConvertSatelliteToNumpyBatch(sat_datapipe, is_hrv=False)
    data = next(iter(sat_datapipe))
    assert BatchKey.satellite_actual in data
    assert BatchKey.satellite_t0_idx in data
    assert BatchKey.hrvsatellite_actual not in data
    assert BatchKey.hrvsatellite_t0_idx not in data


def test_convert_hrvsatellite_to_numpy_batch(sat_datapipe):
    sat_datapipe = AddT0IdxAndSamplePeriodDuration(
        sat_datapipe,
        sample_period_duration=timedelta(minutes=5),
        history_duration=timedelta(minutes=60),
    )
    sat_datapipe = ConvertSatelliteToNumpyBatch(sat_datapipe, is_hrv=True)
    data = next(iter(sat_datapipe))
    assert BatchKey.hrvsatellite_actual in data
    assert BatchKey.hrvsatellite_t0_idx in data
    assert BatchKey.satellite_actual not in data
    assert BatchKey.satellite_t0_idx not in data
