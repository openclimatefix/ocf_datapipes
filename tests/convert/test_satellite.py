from ocf_datapipes.convert import ConvertSatelliteToNumpyBatch


def test_convert_satellite_to_numpy_batch(sat_dp):
    sat_dp = ConvertSatelliteToNumpyBatch(sat_dp, is_hrv=False)
    data = next(iter(sat_dp))
    assert data is not None


def test_convert_hrvsatellite_to_numpy_batch(sat_dp):
    sat_dp = ConvertSatelliteToNumpyBatch(sat_dp, is_hrv=True)
    data = next(iter(sat_dp))
    assert data is not None
