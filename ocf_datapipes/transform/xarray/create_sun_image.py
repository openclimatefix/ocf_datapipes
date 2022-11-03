"""Create image channels of sun position at each pixel"""
from typing import Union

import numpy as np
import pvlib
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.utils.consts import Location
from ocf_datapipes.utils.geospatial import (
    load_geostationary_area_definition_and_transform_latlon,
    load_geostationary_area_definition_and_transform_osgb,
    osgb_to_lat_lon,
)

ELEVATION_MEAN = 37.4
ELEVATION_STD = 12.7
AZIMUTH_MEAN = 177.7
AZIMUTH_STD = 41.7


@functional_datapipe("create_sun_image")
class CreateSunImageIterDataPipe(IterDataPipe):
    """Create Sun image from individual sites"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        normalize: bool = False,
        image_dim: str = "geostationary",
        time_dim: str = "time_utc",
    ):
        """
        Creates a 3D data cube of PV output image x number of timesteps

        TODO Include PV System IDs or something, currently just takes the outputs

        Args:
            source_datapipe: Source datapipe of PV data
            normalize: Whether to normalize based off the image max, or leave raw data
            image_dim: Dimension name for the x and y dimensions
            time_dim: Time dimension name
        """
        self.source_datapipe = source_datapipe
        self.normalize = normalize
        self.x_dim = "x_" + image_dim
        self.y_dim = "y_" + image_dim
        self.time_dim = time_dim

    def __iter__(self) -> xr.DataArray:
        for image_xr in self.source_datapipe:
            # Create empty image to use for the PV Systems, assumes image has x and y coordinates
            sun_image = np.zeros(
                (
                    len(image_xr[self.time_dim]),
                    2,  # Azimuth and elevation
                    len(image_xr[self.y_dim]),
                    len(image_xr[self.x_dim]),
                ),
                dtype=np.float32,
            )
            if "geostationary" in self.x_dim:
                transform_to_latlon = load_geostationary_area_definition_and_transform_latlon(
                    image_xr
                )
                lats, lons = transform_to_latlon(
                    xx=image_xr[self.x_dim].values, yy=image_xr[self.y_dim].values
                )
            else:
                transform_to_latlon = osgb_to_lat_lon
                lats, lons = transform_to_latlon(x=image_xr.x_osgb.values, y=image_xr.y_osgb.values)
            time_utc = image_xr[self.time_dim].values

            # Loop round each example to get the Sun's elevation and azimuth:
            # Go through each time on its own, lat lons still in order of image
            # TODO Make this faster
            # dt = pd.DatetimeIndex(dt)  # pvlib expects a `pd.DatetimeIndex`.
            for y_index, lat in enumerate(lats):
                for x_index, lon in enumerate(lons):
                    solpos = pvlib.solarposition.get_solarposition(
                        time=time_utc,
                        latitude=lat,
                        longitude=lon,
                        # Which `method` to use?
                        # pyephem seemed to be a good mix between speed
                        # and ease but causes segfaults!
                        # nrel_numba doesn't work when using
                        # multiple worker processes.
                        # nrel_c is probably fastest but
                        # requires C code to be manually compiled:
                        # https://midcdmz.nrel.gov/spa/
                    )
                    sun_image[:, 0][y_index][x_index] = solpos["azimuth"]
                    sun_image[:, 1][y_index][x_index] = solpos["elevation"]

            # Normalize.
            if self.normalize:
                sun_image[:, 0] = (sun_image[:, 0] - AZIMUTH_MEAN) / AZIMUTH_STD
                sun_image[:, 1] = (sun_image[:, 1] - ELEVATION_MEAN) / ELEVATION_STD

            # Should return Xarray as in Xarray transforms
            # Same coordinates as the image xarray, so can take that
            sun_image = _create_data_array_from_image(sun_image, image_xr)
            yield sun_image


def _create_data_array_from_image(
    sun_image: np.ndarray,
    image_xr: Union[xr.Dataset, xr.DataArray],
    is_geostationary: bool,
    time_dim: str,
):
    if is_geostationary:
        data_array = xr.DataArray(
            data=sun_image,
            coords=(
                ("time_utc", image_xr[time_dim].values),
                ("channel", ["azimuth", "elevation"]),
                ("y_geostationary", image_xr.y_geostationary.values),
                ("x_geostationary", image_xr.x_geostationary.values),
            ),
            name="sun_image",
        ).astype(np.float32)
    else:
        data_array = xr.DataArray(
            data=sun_image,
            coords=(
                ("time_utc", image_xr[time_dim].values),
                ("channel", ["azimuth", "elevation"]),
                ("y_osgb", image_xr.y_osgb.values),
                ("x_osgb", image_xr.x_osgb.values),
            ),
            name="sun_image",
        ).astype(np.float32)
    data_array.attrs = image_xr.attrs
    return data_array


def _get_idx_of_pixel_closest_to_poi_geostationary(
    xr_data: xr.DataArray,
    center_osgb: Location,
    x_dim_name="x_geostationary",
    y_dim_name="y_geostationary",
) -> Location:
    """
    Return x and y index location of pixel at center of region of interest.

    Args:
        xr_data: Xarray dataset
        center_osgb: Center in OSGB coordinates
        x_dim_name: X dimension name
        y_dim_name: Y dimension name

    Returns:
        Location for the center pixel in geostationary coordinates
    """
    _osgb_to_geostationary = load_geostationary_area_definition_and_transform_osgb(xr_data)
    center_geostationary_tuple = _osgb_to_geostationary(xx=center_osgb.x, yy=center_osgb.y)
    center_geostationary = Location(
        x=center_geostationary_tuple[0], y=center_geostationary_tuple[1]
    )

    # Get the index into x and y nearest to x_center_geostationary and y_center_geostationary:
    x_index_at_center = np.searchsorted(xr_data[x_dim_name].values, center_geostationary.x) - 1
    # y_geostationary is in descending order:
    y_index_at_center = len(xr_data[y_dim_name]) - (
        np.searchsorted(xr_data[y_dim_name].values[::-1], center_geostationary.y) - 1
    )
    return Location(x=x_index_at_center, y=y_index_at_center)
