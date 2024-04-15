from ocf_datapipes.load.sensor import OpenMeteomaticsFromZarr
from ocf_datapipes.config.model import Sensor


def test_open_meteomatics_solar_from_zarr():
    sensor_config = Sensor(
        sensor_filename="tests/data/meteo_solar.zip", sensor_variables=["direct_rad:W"]
    )
    meteo_datapipe = OpenMeteomaticsFromZarr(sensor_config)
    data = next(iter(meteo_datapipe))
    assert data is not None
    assert len(data.station_id) == 20


def test_open_meteomatics_solar_from_zarr():
    sensor_config = Sensor(sensor_filename="tests/data/meteo_wind.zip", sensor_variables=["100v"])
    meteo_datapipe = OpenMeteomaticsFromZarr(sensor_config)
    data = next(iter(meteo_datapipe))
    assert data is not None
    assert len(data.station_id) == 26
