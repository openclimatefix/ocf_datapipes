from numbers import Number
from typing import Union

import numpy as np
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
