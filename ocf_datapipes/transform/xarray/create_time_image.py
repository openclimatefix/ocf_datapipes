"""Create image channels of time values at each pixel"""

from typing import Union

import numpy as np
import xarray as xr
from torch.utils.data import IterDataPipe, functional_datapipe

from ocf_datapipes.utils.utils import trigonometric_datetime_transformation


@functional_datapipe("create_time_image")
class CreateTimeImageIterDataPipe(IterDataPipe):
    """Create Time image"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        image_dim: str = "geostationary",
        time_dim: str = "time_utc",
    ):
        """
        Creates a 3D data cube of Time output image x number of timesteps

        Args:
            source_datapipe: Source datapipe of PV data
            image_dim: Dimension name for the x and y dimensions
            time_dim: Time dimension name
        """
        self.source_datapipe = source_datapipe
        self.x_dim = "x_" + image_dim
        self.y_dim = "y_" + image_dim
        self.time_dim = time_dim

    def __iter__(self) -> xr.DataArray:
        for image_xr in self.source_datapipe:
            # Create empty image to use, assumes image has x and y coordinates
            time_image = _create_time_image(
                image_xr,
                time_dim=self.time_dim,
                output_width_pixels=len(image_xr[self.x_dim]),
                output_height_pixels=len(image_xr[self.y_dim]),
            )
            time_image = _create_data_array_from_image(
                time_image,
                image_xr,
                is_geostationary="x_geostationary" in image_xr.dims,
                time_dim=self.time_dim,
            )
            yield time_image


def _create_time_image(xr_data, time_dim: str, output_height_pixels: int, output_width_pixels: int):
    # Create trig decomposition of datetime values, tiled over output height and width
    datetimes = xr_data[time_dim].values
    trig_decomposition = trigonometric_datetime_transformation(datetimes)
    tiled_data = np.expand_dims(trig_decomposition, (2, 3))
    tiled_data = np.tile(tiled_data, (1, 1, output_height_pixels, output_width_pixels))
    return tiled_data


def _create_data_array_from_image(
    time_image: np.ndarray,
    image_xr: Union[xr.Dataset, xr.DataArray],
    is_geostationary: bool,
    time_dim: str,
):
    if is_geostationary:
        data_array = xr.DataArray(
            data=time_image,
            coords=(
                ("time_utc", image_xr[time_dim].values),
                (
                    "channel",
                    [f"time_channel_{i}" for i in range(time_image.shape[1])],
                ),  # Temp channel names
                ("y_geostationary", image_xr.y_geostationary.values),
                ("x_geostationary", image_xr.x_geostationary.values),
            ),
            name="time_image",
        ).astype(np.float32)
    else:
        data_array = xr.DataArray(
            data=time_image,
            coords=(
                ("time_utc", image_xr[time_dim].values),
                ("channel", [f"time_channel_{i}" for i in range(time_image.shape[1])]),
                ("y_osgb", image_xr.y_osgb.values),
                ("x_osgb", image_xr.x_osgb.values),
            ),
            name="time_image",
        ).astype(np.float32)
    data_array.attrs = image_xr.attrs
    return data_array
