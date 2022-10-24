"""Geospatial functions"""
from datetime import datetime
from numbers import Number
from typing import Union

import numpy as np
import pandas as pd
import pvlib
import pyproj

# OSGB is also called "OSGB 1936 / British National Grid -- United
# Kingdom Ordnance Survey".  OSGB is used in many UK electricity
# system maps, and is used by the UK Met Office UKV model.  OSGB is a
# Transverse Mercator projection, using 'easting' and 'northing'
# coordinates which are in meters.  See https://epsg.io/27700
OSGB36 = 27700

# WGS84 is short for "World Geodetic System 1984", used in GPS. Uses
# latitude and longitude.
WGS84 = 4326


_osgb_to_lat_lon = pyproj.Transformer.from_crs(crs_from=OSGB36, crs_to=WGS84).transform
_lat_lon_to_osgb = pyproj.Transformer.from_crs(crs_from=WGS84, crs_to=OSGB36).transform


def osgb_to_lat_lon(
    x: Union[Number, np.ndarray], y: Union[Number, np.ndarray]
) -> tuple[Union[Number, np.ndarray], Union[Number, np.ndarray]]:
    """Change OSGB coordinates to lat, lon.

    Args:
        x: osgb east-west
        y: osgb north-south
    Return: 2-tuple of latitude (north-south), longitude (east-west).
    """
    return _osgb_to_lat_lon(xx=x, yy=y)


def lat_lon_to_osgb(
    latitude: Union[Number, np.ndarray], longitude: Union[Number, np.ndarray]
) -> tuple[Union[Number, np.ndarray], Union[Number, np.ndarray]]:
    """Change lat, lon coordinates to OSGB.

    Return: 2-tuple of OSGB x, y.
    """
    return _lat_lon_to_osgb(xx=latitude, yy=longitude)


def calculate_azimuth_and_elevation_angle(
    latitude: float, longitude: float, datestamps: list[datetime]
) -> pd.DataFrame:
    """
    Calculation the azimuth angle, and the elevation angle for several datetamps.

    But for one specific osgb location

    More details see:
    https://www.celestis.com/resources/faq/what-are-the-azimuth-and-elevation-of-a-satellite/

    Args:
        latitude: latitude of the pv site
        longitude: longitude of the pv site
        datestamps: list of datestamps to calculate the sun angles. i.e the sun moves from east to
            west in the day.

    Returns: Pandas data frame with the index the same as 'datestamps', with columns of
    "elevation" and "azimuth" that have been calculate.

    """
    # get the solor position
    solpos = pvlib.solarposition.get_solarposition(datestamps, latitude, longitude)

    # extract the information we want
    return solpos[["elevation", "azimuth"]]


def load_geostationary_area_definition_and_transform_osgb(xr_data):
    """
    Loads geostationary area and transformation from OSGB to geostationaery

    Args:
        xr_data: Xarray object with geostationary area

    Returns:
        The transform
    """
    # Only load these if using geostationary projection
    import pyproj
    import pyresample

    area_definition_yaml = xr_data.attrs["area"]
    geostationary_area_definition = pyresample.area_config.load_area_from_string(
        area_definition_yaml
    )
    geostationary_crs = geostationary_area_definition.crs
    osgb_to_geostationary = pyproj.Transformer.from_crs(
        crs_from=OSGB36, crs_to=geostationary_crs
    ).transform
    return osgb_to_geostationary


def load_geostationary_area_definition_and_transform_latlon(xr_data):
    """
    Loads geostationary area and transformation from Latlon to geostationaery

    Args:
        xr_data: Xarray object with geostationary area

    Returns:
        The transform
    """
    # Only load these if using geostationary projection
    import pyproj
    import pyresample

    area_definition_yaml = xr_data.attrs["area"]
    geostationary_area_definition = pyresample.area_config.load_area_from_string(
        area_definition_yaml
    )
    geostationary_crs = geostationary_area_definition.crs
    latlon_to_geostationary = pyproj.Transformer.from_crs(
        crs_from=WGS84, crs_to=geostationary_crs
    ).transform
    return latlon_to_geostationary


def load_geostationary_area_definition_and_transform_to_latlon(xr_data):
    """
    Loads geostationary area and transformation from Latlon to geostationaery

    Args:
        xr_data: Xarray object with geostationary area

    Returns:
        The transform
    """
    # Only load these if using geostationary projection
    import pyproj
    import pyresample

    area_definition_yaml = xr_data.attrs["area"]
    geostationary_area_definition = pyresample.area_config.load_area_from_string(
        area_definition_yaml
    )
    geostationary_crs = geostationary_area_definition.crs
    geostationary_to_latlon = pyproj.Transformer.from_crs(
        crs_to=WGS84, crs_from=geostationary_crs
    ).transform
    return geostationary_to_latlon
