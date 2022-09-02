from ocf_datapipes.load import OpenSatellite
from ocf_datapipes.transform.xarray import ConvertSatelliteToInt8


def test_convert_satellite_to_int8(sat_datapipe):
    sat_datapipe = ConvertSatelliteToInt8(sat_datapipe)
    data = next(iter(sat_datapipe))
    assert data.dtype == "uint8"
