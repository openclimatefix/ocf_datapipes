"""Convert point PV sites to image output"""
import logging
from typing import Union

import numpy as np
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.utils import Zipper

logger = logging.getLogger(__name__)


@functional_datapipe("create_pv_history_image")
class CreatePVHistoryImageIterDataPipe(IterDataPipe):
    """Create pv image from individual sites"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        image_datapipe: IterDataPipe,
        normalize: bool = False,
        image_dim: str = "geostationary",
        always_return_first: bool = False,
        seed=None,
    ):
        """
        Creates a 3D data cube of PV output image x number of timesteps

        This is primarily for site level PV, so single PV inputs are preferred

        Args:
            source_datapipe: Source datapipe of PV data
            image_datapipe: Datapipe emitting images to get the shape from, with coordinates
            normalize: Whether to normalize based off the image max, or leave raw data
            image_dim: Dimension name for the x and y dimensions
            always_return_first: Always return the first image data cube, to save computation
                Only use for if making the image at the beginning of the stack
            seed: Random seed to use if using max_num_pv_systems
        """
        self.source_datapipe = source_datapipe
        self.image_datapipe = image_datapipe
        self.normalize = normalize
        self.x_dim = "x_" + image_dim
        self.y_dim = "y_" + image_dim
        self.rng = np.random.default_rng(seed=seed)
        self.always_return_first = always_return_first

    def __iter__(self) -> xr.DataArray:
        for pv_systems_xr, image_xr in Zipper(self.source_datapipe, self.image_datapipe):
            # Create empty image to use for the PV Systems, assumes image has x and y coordinates
            pv_image = np.zeros(
                (
                    len(pv_systems_xr["time_utc"]),
                    len(image_xr[self.y_dim]),
                    len(image_xr[self.x_dim]),
                ),
                dtype=np.float32,
            )
            # If only one, like chosen before, then use single one
            for time_step in range(len(pv_systems_xr.time_utc.values)):
                # Now go by the timestep to create cube of pv data
                pv_image[time_step:, :] = pv_systems_xr.isel(time_utc=time_step).values

            pv_image = np.nan_to_num(pv_image)

            # Should return Xarray as in Xarray transforms
            # Same coordinates as the image xarray, so can take that
            pv_image = _create_data_array_from_image(
                pv_image, pv_systems_xr, image_xr, image_dim=self.x_dim.split("_")[-1]
            )
            yield pv_image


def _create_data_array_from_image(
    pv_image: np.ndarray,
    pv_systems_xr: Union[xr.Dataset, xr.DataArray],
    image_xr: Union[xr.Dataset, xr.DataArray],
    image_dim: str = "geostationary",
):
    data_array = xr.DataArray(
        data=pv_image,
        coords=(
            ("time_utc", pv_systems_xr.time_utc.values),
            ("y_" + image_dim, image_xr["y_" + image_dim].values),
            ("x_" + image_dim, image_xr["x_" + image_dim].values),
        ),
        name="pv_image",
    ).astype(np.float32)
    data_array.attrs = image_xr.attrs
    return data_array
