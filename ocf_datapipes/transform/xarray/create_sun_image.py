"""Create image channels of sun position at each pixel"""

from typing import Union

import numpy as np
import pvlib
import xarray as xr
from torch.utils.data import IterDataPipe, functional_datapipe

from ocf_datapipes.utils.consts import (
    AZIMUTH_MEAN,
    AZIMUTH_STD,
    ELEVATION_MEAN,
    ELEVATION_STD,
)
from ocf_datapipes.utils.geospatial import (
    geostationary_area_coords_to_lonlat,
    osgb_to_lon_lat,
)


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
                lons, lats = geostationary_area_coords_to_lonlat(
                    x=image_xr[self.x_dim].values, y=image_xr[self.y_dim].values, xr_data=image_xr
                )

            else:
                lons, lats = osgb_to_lon_lat(x=image_xr.x_osgb.values, y=image_xr.y_osgb.values)
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
                    sun_image[:, 0, y_index, x_index] = solpos["azimuth"]
                    sun_image[:, 1, y_index, x_index] = solpos["elevation"]

            # Normalize.
            if self.normalize:
                sun_image[:, 0] = (sun_image[:, 0] - AZIMUTH_MEAN) / AZIMUTH_STD
                sun_image[:, 1] = (sun_image[:, 1] - ELEVATION_MEAN) / ELEVATION_STD

            # Should return Xarray as in Xarray transforms
            # Same coordinates as the image xarray, so can take that
            sun_image = _create_data_array_from_image(
                sun_image,
                image_xr,
                is_geostationary="geostationary" in self.x_dim,
                time_dim=self.time_dim,
            )
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
