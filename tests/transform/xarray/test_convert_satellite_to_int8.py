from ocf_datapipes.load import OpenSatellite
from ocf_datapipes.transform.xarray import ConvertSatelliteToInt8


def test_convert_satellite_to_int8(sat_dp):
    sat_dp = ConvertSatelliteToInt8(sat_dp)
    data = next(iter(sat_dp))
    assert data.dtype == "uint8"
