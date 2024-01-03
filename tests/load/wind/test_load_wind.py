from ocf_datapipes.config.model import Wind, WindFiles
from ocf_datapipes.load import OpenWindFromNetCDF
from datetime import datetime


def test_open_wind_from_nc():
    wind = Wind()
    wind_file = WindFiles(
        wind_filename="tests/data/wind/wind_test_data.nc",
        wind_metadata_filename="tests/data/wind/wind_metadata.csv",
        label="india",
    )
    wind.wind_files_groups = [wind_file]
    wind.start_datetime = datetime(2022, 1, 1)
    wind.end_datetime = datetime(2023, 11, 30)
    wind_datapipe = OpenWindFromNetCDF(wind=wind)
    data = next(iter(wind_datapipe))
    assert data is not None
    assert len(data.wind_system_id) == 1
