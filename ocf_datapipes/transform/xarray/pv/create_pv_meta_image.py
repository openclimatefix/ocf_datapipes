"""Convert point PV sites to image output"""
import logging
from collections import defaultdict
from typing import Union

import numpy as np
import pandas as pd
import pvlib
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.utils import Zipper
from ocf_datapipes.utils.consts import Location
from ocf_datapipes.utils.geospatial import (
    load_geostationary_area_definition_and_transform_osgb,
    load_geostationary_area_definition_and_transform_latlon,
    spatial_coord_type,
)
from ocf_datapipes.utils.utils import searchsorted

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
        image_dim: str = "geostationary",
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
        assert max_num_pv_systems>0
        
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
            
            # Check if the coords are ascending
            x_vals = image_xr[image_x_dim].values
            y_vals = image_xr[image_y_dim].values
            
            x_ascend = (x_vals == np.sort(x_vals)).all()
            y_ascend = (y_vals == np.sort(y_vals)).all()
            
            # Check if coords are descending
            x_descend = (x_vals == np.sort(x_vals)[::-1]).all()
            y_descend = (y_vals == np.sort(y_vals)[::-1]).all()
            
            # Coords must be either ascending or descending order
            assert x_ascend or x_descend
            assert y_ascend or y_descend
            
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
                
                if pv_coords=="osgb" and image_coords=="geostationary":
                    _osgb_to_geostationary = load_geostationary_area_definition_and_transform_osgb(
                        image_xr
                    )
                    pv_y, pv_x = _osgb_to_geostationary(
                        xx=pv_system["x_osgb"].values, 
                        yy=pv_system["y_osgb"].values,
                    )
                elif pv_coords=="lat_lon" and image_coords=="geostationary":
                    _latlon_to_geostationary = (
                        load_geostationary_area_definition_and_transform_latlon(image_xr)
                    )
                    pv_y, pv_x = _latlon_to_geostationary(
                        xx=pv_system["longitude"].values, 
                        yy=pv_system["latitude"].values,
                    )
                    
                else:
                    raise NotImplementedError(
                        f"PV coords of type {pv_coords} and image coords of type {image_coords}"
                    )

                # Check the PV system is within the image
                in_range = (
                    (image_xr[image_x_dim].min() < pv_x < image_xr[image_x_dim].max())
                    and 
                    (image_xr[image_y_dim].min() < pv_y < image_xr[image_y_dim].max())
                )
                
                if not in_range:
                    # Skip the PV system if not inside the image
                    continue
                    
                x_idx = searchsorted(image_xr[image_x_dim], pv_x, assume_ascending=x_ascend)
                y_idx = searchsorted(image_xr[image_y_dim], pv_y, assume_ascending=y_ascend)
                
                # TODO: should we be using an average here? Overwiting will occur as it is
                pv_meta_image[:, y_idx, x_idx] = np.array([pv_system["tilt"], pv_system["orientation"]])

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