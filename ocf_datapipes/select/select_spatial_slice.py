"""Select spatial slices"""
from typing import Union

import numpy as np
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe, Zipper

from ocf_datapipes.utils.consts import Location
from ocf_datapipes.utils.geospatial import load_geostationary_area_definition_and_transform_osgb


@functional_datapipe("select_spatial_slice_pixels")
class SelectSpatialSlicePixelsIterDataPipe(IterDataPipe):
    """Select spatial slice based off pixels from point of interest"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        location_datapipe: IterDataPipe,
        roi_height_pixels: int,
        roi_width_pixels: int,
        y_dim_name: str = "y",
        x_dim_name: str = "x",
    ):
        """
        Select spatial slice based off pixels from point of interest

        Args:
            source_datapipe: Datapipe of Xarray data
            location_datapipe: Location datapipe
            roi_height_pixels: ROI height in pixels
            roi_width_pixels: ROI width in pixels
            y_dim_name: Dimension name for Y
            x_dim_name: Dimension name for X
        """
        self.source_datapipe = source_datapipe
        self.location_datapipe = location_datapipe
        self.roi_height_pixels = roi_height_pixels
        self.roi_width_pixels = roi_width_pixels
        self.y_dim_name = y_dim_name
        self.x_dim_name = x_dim_name

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        for xr_data, location in Zipper(self.source_datapipe, self.location_datapipe):
            if "geostationary" in self.x_dim_name:
                center_idx: Location = _get_idx_of_pixel_closest_to_poi_geostationary(
                    xr_data=xr_data,
                    center_osgb=location,
                    x_dim_name=self.x_dim_name,
                    y_dim_name=self.y_dim_name,
                )
            else:
                center_idx: Location = _get_idx_of_pixel_closest_to_poi(
                    xr_data=xr_data,
                    center_osgb=location,
                    x_dim_name=self.x_dim_name,
                    y_dim_name=self.y_dim_name,
                )

            # Compute the index for left and right:
            half_height = self.roi_height_pixels // 2
            half_width = self.roi_width_pixels // 2

            left_idx = int(center_idx.x - half_width)
            right_idx = int(center_idx.x + half_width)
            top_idx = int(center_idx.y - half_height)
            bottom_idx = int(center_idx.y + half_height)

            # Sanity check!
            assert left_idx >= 0, f"{left_idx=} must be >= 0!"
            data_width_pixels = len(xr_data[self.x_dim_name])
            assert right_idx <= data_width_pixels, f"{right_idx=} must be <= {data_width_pixels=}"
            assert top_idx >= 0, f"{top_idx=} must be >= 0!"
            data_height_pixels = len(xr_data[self.y_dim_name])
            assert (
                bottom_idx <= data_height_pixels
            ), f"{bottom_idx=} must be <= {data_height_pixels=}"

            selected = xr_data.isel(
                {
                    self.x_dim_name: slice(left_idx, right_idx),
                    self.y_dim_name: slice(top_idx, bottom_idx),
                }
            )
            yield selected


@functional_datapipe("select_spatial_slice_meters")
class SelectSpatialSliceMetersIterDataPipe(IterDataPipe):
    """Select spatial slice based off meters from point of interest

    Currently assumes that there is pv_system_id to use isel on
    """

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        location_datapipe: IterDataPipe,
        roi_height_meters: int,
        roi_width_meters: int,
        dim_name: str = "pv_system_id",
    ):
        """
        Select spatial slice based off pixels from point of interest

        Args:
            source_datapipe: Datapipe of Xarray data
            location_datapipe: Location datapipe
            roi_height_meters: ROI height in meters
            roi_width_meters: ROI width in meters
            dim_name: Dimension name to select for ID
        """
        self.source_datapipe = source_datapipe
        self.location_datapipe = location_datapipe
        self.roi_height_meters = roi_height_meters
        self.roi_width_meters = roi_width_meters
        self.dim_name = dim_name

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        for xr_data, location in Zipper(self.source_datapipe, self.location_datapipe):
            # Compute the index for left and right:
            half_height = self.roi_height_meters // 2
            half_width = self.roi_width_meters // 2

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

            selected = xr_data.isel({self.dim_name: id_mask})
            yield selected


def _get_idx_of_pixel_closest_to_poi(
    xr_data: xr.DataArray, center_osgb: Location, y_dim_name: str = "y", x_dim_name: str = "x"
) -> Location:
    """
    Return x and y index location of pixel at center of region of interest.

    Args:
        xr_data: Xarray dataset
        center_osgb: Center in OSGB coordinates
        y_dim_name: Y dimension name
        x_dim_name: X dimension name

    Returns:
        The Location for the center pixel
    """
    y_index = xr_data.get_index(y_dim_name)
    x_index = xr_data.get_index(x_dim_name)
    return Location(
        y=y_index.get_indexer([float(center_osgb.y)], method="nearest")[0],
        x=x_index.get_indexer([float(center_osgb.x)], method="nearest")[0],
    )


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
