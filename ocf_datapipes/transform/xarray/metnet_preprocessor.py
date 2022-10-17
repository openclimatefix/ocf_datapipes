"""Preprocessing for MetNet-type inputs"""
from typing import List

import numpy as np
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe, Zipper


@functional_datapipe("preprocess_metnet")
class PreProcessMetNetIterDataPipe(IterDataPipe):
    """Preprocess set of Xarray datasets similar to MetNet-1"""

    def __init__(
        self,
        source_datapipes: List[IterDataPipe],
        location_datapipe: IterDataPipe,
        context_width: float,
        context_height: float,
        center_width: float,
        center_height: float,
        output_height_pixels: int,
        output_width_pixels: int,
    ):
        """

        Processes set of Xarray datasets similar to MetNet

        In terms of taking all available source datapipes:
        1. selecting the same context area of interest
        2. Creating a center crop of the center_height, center_width
        3. Downsampling the context area of interest to the same shape as the center crop
        4. Stacking those context images on the center crop.

        This would be designed originally for NWP+Satellite+Topographic data sources.
        To add the PV power for lots of sites, the PV power would
        need to be able to be on a grid for the context/center
        crops and then for the downsample

        This also appends Lat/Lon coordinates to the stack,
         and returns a new Numpy array with the stacked data

        TODO Could also add the national PV as a set of Layers, so also GSP input

        Args:
            source_datapipes: Datapipes that emit xarray datasets
                with latitude/longitude coordinates included
            location_datapipe: Datapipe emitting location coordinate for center of example
            context_width: Width of the context area
            context_height: Height of the context area
            center_width: Center width of the area of interest
            center_height: Center height of the area of interest
            output_height_pixels: Output height in pixels
            output_width_pixels: Output width in pixels
        """
        self.source_datapipes = source_datapipes
        self.location_datapipe = location_datapipe
        self.context_width = context_width
        self.context_height = context_height
        self.center_width = center_width
        self.center_height = center_height

    def __iter__(self) -> np.ndarray:
        for xr_datas, location in Zipper(Zipper(*self.source_datapipes), self.location_datapipe):
            # TODO Use the Lat/Long coordinates of the center array for the lat/lon stuff
            centers = []
            contexts = []

            for xr_data in xr_datas:
                xr_context: xr.Dataset = _get_spatial_crop(xr_data, location=location,
                                               roi_width_meters=self.context_width,
                                               roi_height_meters=self.context_height,
                                               dim_name="data")
                xr_center: xr.Dataset = _get_spatial_crop(xr_data, location=location,
                                               roi_width_meters=self.center_width,
                                               roi_height_meters=self.center_height,
                                               dim_name="data")
                # TODO Resample here before stacking
                xr_center = xr_center.coarsen(y=3).mean().coarsen(x=3).mean()
                xr_context = xr_context.coarsen(y=3).mean().coarsen(x=3).mean()
                centers.append(xr_center)
                contexts.append(xr_context)
            stacked_data = np.stack([*centers, *contexts], dim=-1)
            yield stacked_data


def _get_spatial_crop(xr_data, location, roi_height_meters, roi_width_meters, dim_name):
    # Compute the index for left and right:
    half_height = roi_height_meters // 2
    half_width = roi_width_meters // 2

    left = location.x - half_width
    right = location.x + half_width
    bottom = location.y - half_height
    top = location.y + half_height
    # Select data in the region of interest:
    id_mask = (
        (left <= xr_data.x_osgb)
        & (xr_data.x_osgb <= right)
        & (xr_data.y_osgb <= top)
        & (bottom <= xr_data.y_osgb)
    )

    selected = xr_data.isel({dim_name: id_mask})
    return selected
