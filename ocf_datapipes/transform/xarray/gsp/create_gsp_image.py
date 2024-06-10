"""Convert point PV sites to image output"""

import logging
from typing import Union

import numpy as np
import xarray as xr
from torch.utils.data import IterDataPipe, functional_datapipe

from ocf_datapipes.utils import Zipper

logger = logging.getLogger(__name__)


@functional_datapipe("create_gsp_image")
class CreateGSPImageIterDataPipe(IterDataPipe):
    """Create GSP image from individual sites"""

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
        Creates a 3D data cube of GSP output image x number of timesteps

        This is primarily for national GSP, so single GSP inputs are preferred

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
        for gsp_systems_xr, image_xr in Zipper(self.source_datapipe, self.image_datapipe):
            # Create empty image to use for the PV Systems, assumes image has x and y coordinates
            pv_image = np.zeros(
                (
                    len(gsp_systems_xr["time_utc"]),
                    len(image_xr[self.y_dim]),
                    len(image_xr[self.x_dim]),
                ),
                dtype=np.float32,
            )
            for i, gsp_system_id in enumerate(gsp_systems_xr["gsp_id"]):
                gsp_system = gsp_systems_xr.sel(gsp_id=gsp_system_id)
                for time_step in range(len(gsp_system.time_utc.values)):
                    # Now go by the timestep to create cube of GSP data
                    pv_image[time_step:, :] = gsp_system.isel(time_utc=time_step).values

            pv_image = np.nan_to_num(pv_image)

            # Should return Xarray as in Xarray transforms
            # Same coordinates as the image xarray, so can take that
            pv_image = _create_data_array_from_image(
                pv_image, gsp_systems_xr, image_xr, image_dim=self.x_dim.split("_")[-1]
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
        name="gsp_image",
    ).astype(np.float32)
    data_array.attrs = image_xr.attrs
    return data_array
