"""Preprocessing for MetNet-type inputs"""
from typing import List

import numpy as np
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe, Zipper

from ocf_datapipes.utils.geospatial import load_geostationary_area_definition_and_transform_osgb


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
        self.output_height_pixels = output_height_pixels
        self.output_width_pixels = output_width_pixels

    def __iter__(self) -> np.ndarray:
        for xr_datas, location in Zipper(Zipper(*self.source_datapipes), self.location_datapipe):
            # TODO Use the Lat/Long coordinates of the center array for the lat/lon stuff
            centers = []
            contexts = []
            for xr_data in xr_datas:
                xr_context: xr.Dataset = _get_spatial_crop(
                    xr_data,
                    location=location,
                    roi_width_meters=self.context_width,
                    roi_height_meters=self.context_height,
                )
                xr_center: xr.Dataset = _get_spatial_crop(
                    xr_data,
                    location=location,
                    roi_width_meters=self.center_width,
                    roi_height_meters=self.center_height,
                )
                # Resamples to the same number of pixels for both center and contexts
                xr_center = _resample_to_pixel_size(
                    xr_center, self.output_height_pixels, self.output_width_pixels
                )
                xr_context = _resample_to_pixel_size(
                    xr_context, self.output_height_pixels, self.output_width_pixels
                )
                xr_center = xr_center.to_numpy()
                xr_context = xr_context.to_numpy()
                if len(xr_center.shape) == 3:  # Need to add channel dimension
                    xr_center = np.expand_dims(xr_center, axis=1)
                    xr_context = np.expand_dims(xr_context, axis=1)
                centers.append(xr_center)
                contexts.append(xr_context)
            # Pad out time dimension to be the same, using the largest one
            # All should have 4 dimensions at this point
            for i in range(len(centers)):
                centers[i] = np.pad(
                    centers[i],
                    pad_width=(
                        (0, np.max([c.shape[0] for c in centers]) - centers[i].shape[0]),
                        (0, 0),
                        (0, 0),
                        (0, 0),
                    ),
                    mode="constant",
                    constant_values=0.0,
                )
                contexts[i] = np.pad(
                    contexts[i],
                    pad_width=(
                        (0, np.max([c.shape[0] for c in contexts]) - contexts[i].shape[0]),
                        (0, 0),
                        (0, 0),
                        (0, 0),
                    ),
                    mode="constant",
                    constant_values=0.0,
                )
            stacked_data = np.concatenate([*centers, *contexts], axis=1)
            yield stacked_data


def _get_spatial_crop(xr_data, location, roi_height_meters: int, roi_width_meters: int):
    # Compute the index for left and right:
    half_height = roi_height_meters // 2
    half_width = roi_width_meters // 2

    left = location.x - half_width
    right = location.x + half_width
    bottom = location.y - half_height
    top = location.y + half_height
    if "x_geostationary" in xr_data.coords:
        # Convert to geostationary edges
        _osgb_to_geostationary = load_geostationary_area_definition_and_transform_osgb(xr_data)
        left, bottom = _osgb_to_geostationary(xx=left, yy=bottom)
        right, top = _osgb_to_geostationary(xx=right, yy=top)
        x_mask = (left <= xr_data.x_geostationary) & (xr_data.x_geostationary <= right)
        y_mask = (xr_data.y_geostationary <= top) & (  # Y is flipped
            bottom <= xr_data.y_geostationary
        )
        selected = xr_data.isel(x_geostationary=x_mask, y_geostationary=y_mask)
    elif "x" in xr_data.coords:
        _osgb_to_geostationary = load_geostationary_area_definition_and_transform_osgb(xr_data)
        left, bottom = _osgb_to_geostationary(xx=left, yy=bottom)
        right, top = _osgb_to_geostationary(xx=right, yy=top)
        x_mask = (left <= xr_data.x) & (xr_data.x <= right)
        y_mask = (xr_data.y <= top) & (bottom <= xr_data.y)  # Y is flipped
        selected = xr_data.isel(x=x_mask, y=y_mask)
    else:
        # Select data in the region of interest:
        x_mask = (left <= xr_data.x_osgb) & (xr_data.x_osgb <= right)
        y_mask = (xr_data.y_osgb <= top) & (bottom <= xr_data.y_osgb)
        selected = xr_data.isel(x_osgb=x_mask, y_osgb=y_mask)

    return selected


def _resample_to_pixel_size(xr_data, height_pixels, width_pixels) -> np.ndarray:
    if "x_geostationary" in xr_data.coords:
        x_coords = xr_data["x_geostationary"].values
        y_coords = xr_data["y_geostationary"].values
    elif "x" in xr_data.coords:
        x_coords = xr_data["x"].values
        y_coords = xr_data["y"].values
    else:
        x_coords = xr_data["x_osgb"].values
        y_coords = xr_data["y_osgb"].values
    # Resample down to the number of pixels wanted
    x_coords = np.linspace(x_coords[0], x_coords[-1], num=width_pixels)
    y_coords = np.linspace(y_coords[0], y_coords[-1], num=height_pixels)
    if "x_geostationary" in xr_data.coords:
        xr_data = xr_data.interp(
            x_geostationary=x_coords, y_geostationary=y_coords, method="linear"
        )
    elif "x" in xr_data.coords:
        xr_data = xr_data.interp(x=x_coords, y=y_coords, method="linear")
    else:
        xr_data = xr_data.interp(x_osgb=x_coords, y_osgb=y_coords, method="linear")
    # Extract just the data now
    return xr_data
