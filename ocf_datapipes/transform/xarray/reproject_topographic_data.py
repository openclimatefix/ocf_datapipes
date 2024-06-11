"""Reproject Topographic data to OSGB"""

from pathlib import Path
from typing import Union

import numpy as np
import pyproj
import pyresample
import xarray as xr
from torch.utils.data import IterDataPipe, functional_datapipe

try:
    import cartopy.crs as ccrs

    _has_cartopy = True
except ImportError:
    _has_cartopy = False


@functional_datapipe("reproject_topography")
class ReprojectTopographyIterDataPipe(IterDataPipe):
    """Reproject Topographic data to OSGB"""

    def __init__(self, topo_datapipe: Union[Path, str]):
        """
        Reproject topo data to OSGB

        Args:
            topo_datapipe: Datapipe emitting topographic data
        """
        self.topo_datapipe = topo_datapipe

    def __iter__(self) -> xr.DataArray:
        """Reproject topographic data"""
        for topo in self.topo_datapipe:
            # Select Western Europe:
            topo = topo.sel(x_osgb=slice(-300_000, 1_500_000), y_osgb=slice(1_300_000, -800_000))

            topo = reproject_topo_data_from_osgb_to_geostationary(topo)
            topo = topo.fillna(0)

            while True:
                yield topo


def reproject_topo_data_from_osgb_to_geostationary(topo: xr.DataArray) -> xr.DataArray:
    """
    Reproject topographic data from osgb to geostationary

    Args:
        topo: Topographic Xarray DataArray

    Returns:
        Reprojected Topographic Xarray DataArray
    """
    topo_osgb_area_def = _get_topo_osgb_area_def(topo)
    topo_geostationary_area_def = _get_topo_geostationary_area_def(topo)
    topo_image = pyresample.image.ImageContainerQuick(topo.values, topo_osgb_area_def)
    topo_image_resampled = topo_image.resample(topo_geostationary_area_def)
    topo_dataarray = _get_data_array_of_resampled_topo_image(topo_image_resampled)
    # topo_dataarray.attrs["area"] = str(topo_geostationary_area_def)
    return topo_dataarray


def _get_topo_osgb_area_def(topo: xr.DataArray) -> pyresample.geometry.AreaDefinition:
    """
    Get the area definition for resampling for OSGB

    Args:
        topo: Topographic Xarray DataArray

    Returns:
        AreaDefinition for OSGB
    """
    # Get AreaDefinition of the OSGB topographical data:
    if not _has_cartopy:
        raise Exception("Please install `cartopy` before using ReprojectTopography")
    osgb = ccrs.OSGB(approx=False)
    return pyresample.create_area_def(
        area_id="OSGB",
        projection=osgb.proj4_params,
        shape=topo.shape,  # y, x
        area_extent=(
            topo.x_osgb[0].item(),  # lower_left_x
            topo.y_osgb[-1].item(),  # lower_left_y
            topo.x_osgb[-1].item(),  # upper_right_x
            topo.y_osgb[0].item(),  # upper_right_y
        ),
    )


def _get_topo_geostationary_area_def(topo: xr.DataArray) -> pyresample.geometry.AreaDefinition:
    """
    Get the geostationary area definition for Topographic data

    Args:
        topo: Topographic Xarray DataArray

    Returns:
        AreaDefinition for geostationary area of RSS imagery
    """
    # Get the geostationary boundaries of the topo data:
    OSGB_EPSG_CODE = 27700
    GEOSTATIONARY_PROJ = {
        "proj": "geos",
        "lon_0": 9.5,
        "h": 35785831,
        "x_0": 0,
        "y_0": 0,
        "a": 6378169,
        "rf": 295.488065897014,
        "no_defs": None,
        "type": "crs",
    }
    osgb_to_geostationary = pyproj.Transformer.from_crs(
        crs_from=OSGB_EPSG_CODE, crs_to=GEOSTATIONARY_PROJ
    ).transform
    lower_left_geos = osgb_to_geostationary(xx=topo.x_osgb[0], yy=topo.y_osgb[-1])
    upper_right_geos = osgb_to_geostationary(xx=topo.x_osgb[-1], yy=topo.y_osgb[0])
    shape = (topo.shape[0] * 2, topo.shape[1] * 2)  # Oversample to make sure we don't loose info.
    return pyresample.create_area_def(
        area_id="msg_seviri_rss_1km",
        projection=GEOSTATIONARY_PROJ,
        shape=shape,
        # lower_left_x, lower_left_y, upper_right_x, upper_right_y:
        area_extent=lower_left_geos + upper_right_geos,
    )


def _get_data_array_of_resampled_topo_image(
    topo_image_resampled: pyresample.image.ImageContainer,
) -> xr.DataArray:
    """
    Put resampled data into a DataArray

    Args:
        topo_image_resampled: Resampled data

    Returns:
        Xarray DataArray containing the resampled data
    """
    (
        lower_left_x,
        lower_left_y,
        upper_right_x,
        upper_right_y,
    ) = topo_image_resampled.geo_def.area_extent
    return xr.DataArray(
        topo_image_resampled.image_data,
        coords=(
            (
                "y",
                np.linspace(
                    start=upper_right_y, stop=lower_left_y, num=topo_image_resampled.shape[0]
                ),
            ),
            (
                "x",
                np.linspace(
                    start=lower_left_x, stop=upper_right_x, num=topo_image_resampled.shape[1]
                ),
            ),
        ),
    )
