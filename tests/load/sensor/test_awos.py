from ocf_datapipes.load.sensor import OpenAWOSFromNetCDF
from ocf_datapipes.config.model import Sensor


def test_open_awos_from_netcdf():
    sensor_config = Sensor(sensor_filename="tests/data/sensor/awos.nc")
    awos_datapipe = OpenAWOSFromNetCDF(sensor_config)
    data = next(iter(awos_datapipe))
    assert data is not None
    assert len(data.station_coord) == 123

