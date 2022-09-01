"""Convert coordinates Datapipes"""
from typing import Union

import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.utils.geospatial import (
    lat_lon_to_osgb,
    load_geostationary_area_definition_and_transform_latlon,
    osgb_to_lat_lon,
)


@functional_datapipe("convert_latlon_to_osgb")
class ConvertLatLonToOSGBIterDataPipe(IterDataPipe):
    """Convert from Lat/Lon object to OSGB"""

    def __init__(self, source_datapipe: IterDataPipe):
        """
        Convert from Lat/Lon to OSGB

        Args:
            source_datapipe: Datapipe emitting Xarray objects with latitude and longitude data
        """
        self.source_datapipe = source_datapipe

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        """Convert from Lat/Lon to OSGB"""
        for xr_data in self.source_datapipe:
            xr_data["x_osgb"], xr_data["y_osgb"] = lat_lon_to_osgb(
                latitude=xr_data["latitude"], longitude=xr_data["longitude"]
            )
            yield xr_data


@functional_datapipe("convert_osgb_to_latlon")
class ConvertOSGBToLatLonIterDataPipe(IterDataPipe):
    """Convert from OSGB to Lat/Lon"""

    def __init__(self, source_datapipe: IterDataPipe):
        """
        Convert from OSGB to Lat/Lon

        Args:
            source_datapipe: Datapipe emitting Xarray objects with OSGB data
        """
        self.source_datapipe = source_datapipe

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        """Convert and add lat/lon to Xarray object"""
        for xr_data in self.source_datapipe:
            xr_data["latitude"], xr_data["longitude"] = osgb_to_lat_lon(
                x=xr_data["x_osgb"], y=xr_data["y_osgb"]
            )
            yield xr_data


@functional_datapipe("convert_geostationary_to_latlon")
class ConvertGeostationaryToLatLonIterDataPipe(IterDataPipe):
    """Convert from geostationary to Lat/Lon points"""

    def __init__(self, source_datapipe: IterDataPipe):
        """
        Convert from Geostationary to Lat/Lon points and add to Xarray object

        Args:
            source_datapipe: Datapipe emitting Xarray object with geostationary points
        """
        self.source_datapipe = source_datapipe

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        """Convert from geostationary to Lat/Lon and yield the Xarray object"""
        for xr_data in self.source_datapipe:
            transform = load_geostationary_area_definition_and_transform_latlon(xr_data)
            xr_data["latitude"], xr_data["longitude"] = transform(
                xx=xr_data["x_geostationary"], yy=xr_data["y_geostationary"]
            )
            yield xr_data
