"""Convert coordinates Datapipes"""

from typing import Union

import xarray as xr
from torch.utils.data import IterDataPipe, functional_datapipe

from ocf_datapipes.utils.geospatial import (
    geostationary_area_coords_to_lonlat,
    lon_lat_to_osgb,
    osgb_to_lon_lat,
)


@functional_datapipe("convert_lonlat_to_osgb")
class ConvertLonLatToOSGBIterDataPipe(IterDataPipe):
    """Convert from Lon/Lat object to OSGB"""

    def __init__(self, source_datapipe: IterDataPipe):
        """
        Convert from Lon/Lat to OSGB

        Args:
            source_datapipe: Datapipe emitting Xarray objects with latitude and longitude data
        """
        self.source_datapipe = source_datapipe

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        """Convert from Lon/Lat to OSGB"""
        for xr_data in self.source_datapipe:
            xr_data["x_osgb"], xr_data["y_osgb"] = lon_lat_to_osgb(
                longitude=xr_data["longitude"],
                latitude=xr_data["latitude"],
            )
            yield xr_data


@functional_datapipe("convert_osgb_to_lonlat")
class ConvertOSGBToLonLatIterDataPipe(IterDataPipe):
    """Convert from OSGB to Lon/Lat"""

    def __init__(self, source_datapipe: IterDataPipe):
        """
        Convert from OSGB to lon-lat coordinates

        Args:
            source_datapipe: Datapipe emitting Xarray objects with OSGB data
        """
        self.source_datapipe = source_datapipe

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        """Convert and add lat/lon to Xarray object"""
        for xr_data in self.source_datapipe:
            xr_data["longitude"], xr_data["latitude"] = osgb_to_lon_lat(
                x=xr_data["x_osgb"], y=xr_data["y_osgb"]
            )
            yield xr_data


@functional_datapipe("convert_geostationary_to_lonlat")
class ConvertGeostationaryToLonLatIterDataPipe(IterDataPipe):
    """Convert from geostationary to Lon/Lat points"""

    def __init__(self, source_datapipe: IterDataPipe):
        """
        Convert from Geostationary to Lon/Lat points and add to Xarray object

        Args:
            source_datapipe: Datapipe emitting Xarray object with geostationary points
        """
        self.source_datapipe = source_datapipe

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        """Convert from geostationary to Lon/Lat and yield the Xarray object"""
        for xr_data in self.source_datapipe:
            xr_data["longitude"], xr_data["latitude"] = geostationary_area_coords_to_lonlat(
                x=xr_data["x_geostationary"],
                y=xr_data["y_geostationary"],
                xr_data=xr_data,
            )

            yield xr_data
