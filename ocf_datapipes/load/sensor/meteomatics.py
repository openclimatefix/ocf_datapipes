"""Load ASOS data from local files for training/inference"""
import logging

import fsspec
import xarray as xr
from torch.utils.data import IterDataPipe, functional_datapipe
import ocf_blosc2

from ocf_datapipes.config.model import Sensor

_log = logging.getLogger(__name__)


@functional_datapipe("OpenMeteomatics")
class OpenMeteomaticsFromNetCDFIterDataPipe(IterDataPipe):
    """OpenMeteomaticsFromNetCDFIterDataPipe"""

    def __init__(
        self,
        sensor: Sensor,
    ):
        """
        Datapipe to load Meteomatics point data

        Args:
            sensor: Sensor configuration
        """
        super().__init__()
        self.sensor = sensor
        self.filename = self.sensor.sensor_filename
        self.variables = list(self.sensor.sensor_variables)

    def __iter__(self):
        with fsspec.open(self.filename, "rb") as f:
            ds = xr.open_mfdataset(f, engine="zarr", combine="nested", concat_dim="validdate")
            ds = ds.rename({"validdate": "time_utc"})
            ds = ds.sortby("time_utc")
            # Get all unique combinations of lat/lon
            lats = ds.lat.values
            lons = ds.lon.values
            unique_lat_lon_sets = set()
            for i, lat in enumerate(lats):
                unique_lat_lon_sets.add((lat, lons[i]))

            # Sort the unique lat/lon sets by lat then lon
            unique_lat_lon_sets = sorted(unique_lat_lon_sets, key=lambda x: (x[0], x[1]))
            # Set a station_id for each lat/lon
            station_id = 0
            dses = []
            for lat, lon in unique_lat_lon_sets:
                # Select all points whose latitude = lat and longitude = lon
                # Use where clauses to select them
                subset = ds.where((ds.lat == lat) & (ds.lon == lon), drop=True)
                dses.append(subset)
                dses[-1] = dses[-1].assign_coords(station_id=station_id)
                station_id += 1
            # Now combine all the datasets into one
            ds = xr.concat(dses, dim="station_id")
            ds = ds[self.variables].to_array()
        while True:
            yield ds


if __name__ == "__main__":
    filename = "/home/jacob/nw_india/wind_archive_2023-03.zarr.zip"
    ds = xr.open_zarr(filename)
    ds = ds.rename({"validdate": "time_utc"})
    ds = ds.sortby("time_utc")
    print(ds)
    # Get all unique combinations of lat/lon
    lats = ds.lat.values
    lons = ds.lon.values
    unique_lat_lon_sets = set()
    for i, lat in enumerate(lats):
        unique_lat_lon_sets.add((lat, lons[i]))
    print(unique_lat_lon_sets)
    print(len(unique_lat_lon_sets))
    # Set a station_id for each lat/lon
    station_id = 0
    dses = []
    for lat, lon in unique_lat_lon_sets:
        # Select all points whose latitude = lat and longitude = lon
        # Use where clauses to select them
        subset = ds.where((ds.lat == lat) & (ds.lon == lon), drop=True)
        dses.append(subset)
        dses[-1] = dses[-1].assign_coords(station_id=station_id)
        station_id += 1
    # Now combine all the datasets into one
    ds = xr.concat(dses, dim="station_id")
    print(ds)
    exit()
    sensor = Sensor(
        sensor_filename=filename,
        sensor_variables=["ws"],
    )
    data = OpenMeteomaticsFromNetCDFIterDataPipe(sensor)
    for d in data:
        print(d)
        break
