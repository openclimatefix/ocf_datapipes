"""Convert point PV sites to image output"""

import logging

import numpy as np
import xarray as xr
from torch.utils.data import IterDataPipe, functional_datapipe

from ocf_datapipes.utils import Zipper
from ocf_datapipes.utils.geospatial import (
    lon_lat_to_geostationary_area_coords,
    osgb_to_geostationary_area_coords,
    spatial_coord_type,
)

logger = logging.getLogger(__name__)

MAX_TILT = 90.0
MAX_ORIENTATION = 360.0


@functional_datapipe("create_pv_metadata_image")
class CreatePVMetadataImageIterDataPipe(IterDataPipe):
    """Create PV Metadata image from individual sites"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        image_datapipe: IterDataPipe,
        normalize: bool = False,
        max_num_pv_systems: int = np.inf,
        always_return_first: bool = False,
        seed: int = None,
    ):
        """
        Creates a 2D data cube  of PV Metadata output image

        The metadata is orientation + tilt

        Args:
            source_datapipe: Source datapipe of PV data
            image_datapipe: Datapipe emitting images to get the shape from, with coordinates
            normalize: Whether to normalize based off the image max, or leave raw data
            max_num_pv_systems: Max number of PV systems to consider to generate entire image
            always_return_first: Always return the first image data cube, to save computation
                Only use for if making the image at the beginning of the stack
            seed: Random seed to use if using max_num_pv_systems
        """
        assert max_num_pv_systems > 0

        self.source_datapipe = source_datapipe
        self.image_datapipe = image_datapipe
        self.normalize = normalize
        self.max_num_pv_systems = max_num_pv_systems
        self.rng = np.random.default_rng(seed=seed)
        self.always_return_first = always_return_first

    def __iter__(self) -> xr.DataArray:
        for pv_systems_xr, image_xr in Zipper(self.source_datapipe, self.image_datapipe):
            pv_coords, pv_x_dim, pv_y_dim = spatial_coord_type(pv_systems_xr)
            image_coords, image_x_dim, image_y_dim = spatial_coord_type(image_xr)

            # Randomly sample systems if too many
            if self.max_num_pv_systems <= len(pv_systems_xr.pv_system_id.values):
                subset_of_pv_system_ids = self.rng.choice(
                    pv_systems_xr.pv_system_id,
                    size=self.max_num_pv_systems,
                    replace=False,
                )
                pv_systems_xr = pv_systems_xr.sel(pv_system_id=subset_of_pv_system_ids)

            # Create empty image to use for the PV Systems, assumes image has x and y coordinates
            pv_meta_image = np.zeros(
                (2, len(image_xr[image_y_dim]), len(image_xr[image_x_dim])),
                dtype=np.float32,
            )

            for i, pv_system_id in enumerate(pv_systems_xr["pv_system_id"]):
                # went for isel incase there is a duplicated pv_system_id
                pv_system = pv_systems_xr.isel(pv_system_id=i)

                if pv_coords == "osgb" and image_coords == "geostationary":
                    pv_x, pv_y = osgb_to_geostationary_area_coords(
                        x=pv_system["x_osgb"].values,
                        y=pv_system["y_osgb"].values,
                        xr_data=image_xr,
                    )
                elif pv_coords == "lon_lat" and image_coords == "geostationary":
                    pv_x, pv_y = lon_lat_to_geostationary_area_coords(
                        x=pv_system["longitude"].values,
                        y=pv_system["latitude"].values,
                        xr_data=image_xr,
                    )

                else:
                    raise NotImplementedError(
                        f"PV coords of type {pv_coords} and image coords of type {image_coords}"
                    )

                # Check the PV system is within the image
                in_range = (image_xr[image_x_dim].min() < pv_x < image_xr[image_x_dim].max()) and (
                    image_xr[image_y_dim].min() < pv_y < image_xr[image_y_dim].max()
                )

                if not in_range:
                    # Skip the PV system if not inside the image
                    continue

                x_idx = image_xr.get_index(image_x_dim).get_indexer([pv_x], method="nearest")[0]
                y_idx = image_xr.get_index(image_y_dim).get_indexer([pv_y], method="nearest")[0]

                # TODO: should we be using an average here? Overwiting will occur as it is
                pv_meta_image[:, y_idx, x_idx] = np.array(
                    [pv_system["tilt"], pv_system["orientation"]]
                )

            if self.normalize:
                pv_meta_image = pv_meta_image / np.array([MAX_TILT, MAX_ORIENTATION])[:, None, None]

            # Should return xarray
            # Same coordinates as the image xarray, so can take that
            pv_meta_image_xr = xr.DataArray(
                data=pv_meta_image,
                coords=(
                    # TODO fix coordinate name?
                    ("time_utc", ["tilt", "orientation"]),
                    (image_y_dim, image_xr[image_y_dim].values),
                    (image_x_dim, image_xr[image_x_dim].values),
                ),
                name="pv_meta_image",
            )
            pv_meta_image_xr.attrs = image_xr.attrs

            yield pv_meta_image_xr
