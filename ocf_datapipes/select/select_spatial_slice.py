"""Select spatial slices"""
import logging
from typing import Optional, Union

import numpy as np
import xarray as xr
from scipy.spatial import KDTree
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.utils.consts import Location
from ocf_datapipes.utils.geospatial import (
    lat_lon_to_osgb,
    load_geostationary_area_definition_and_transform_latlon,
    load_geostationary_area_definition_and_transform_osgb,
    move_lat_lon_by_meters,
    osgb_to_lat_lon,
)

logger = logging.getLogger(__name__)


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
        location_idx_name: Optional[str] = None,
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
            location_idx_name: Name for location index of unstructured grid data,
                None if not relevant
        """
        self.source_datapipe = source_datapipe
        self.location_datapipe = location_datapipe
        self.roi_height_pixels = roi_height_pixels
        self.roi_width_pixels = roi_width_pixels
        self.y_dim_name = y_dim_name
        self.x_dim_name = x_dim_name
        self.location_idx_name = location_idx_name

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        for xr_data, location in self.source_datapipe.zip_ocf(self.location_datapipe):
            logger.debug("Selecting spatial slice with pixels")
            if self.location_idx_name is not None:
                selected = _get_points_from_unstructured_grids(
                    xr_data=xr_data,
                    location=location,
                    x_dim_name=self.x_dim_name,
                    y_dim_name=self.y_dim_name,
                    location_idx_name=self.location_idx_name,
                    num_points=self.roi_width_pixels * self.roi_height_pixels,
                )
                yield selected

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
                    location=location,
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
        dim_name: Optional[str] = "pv_system_id",
        y_dim_name: str = "y_osgb",
        x_dim_name: str = "x_osgb",
    ):
        """
        Select spatial slice based off pixels from point of interest

        Args:
            source_datapipe: Datapipe of Xarray data
            location_datapipe: Location datapipe
            roi_height_meters: ROI height in meters
            roi_width_meters: ROI width in meters
            dim_name: Dimension name to select for ID, None for coordinates
            y_dim_name: the y dimension name, this is so we can switch between osgb and lat,lon
            x_dim_name: the x dimension name, this is so we can switch between osgb and lat,lon
        """
        self.source_datapipe = source_datapipe
        self.location_datapipe = location_datapipe
        self.roi_height_meters = roi_height_meters
        self.roi_width_meters = roi_width_meters
        self.dim_name = dim_name
        self.y_dim_name = y_dim_name
        self.x_dim_name = x_dim_name

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        for xr_data, location in self.source_datapipe.zip_ocf(self.location_datapipe):
            # Compute the index for left and right:
            logger.debug("Getting Spatial Slice Meters")

            half_height = self.roi_height_meters // 2
            half_width = self.roi_width_meters // 2
            if location.coordinate_system == "lat_lon":
                top, right = move_lat_lon_by_meters(
                    location.latitude, location.longitude, half_height, half_width
                )
                bottom, left = move_lat_lon_by_meters(
                    location.latitude, location.longitude, -half_height, -half_width
                )
            elif location.coordinate_system == "osgb":
                left = location.x - half_width
                right = location.x + half_width
                bottom = location.y - half_height
                top = location.y + half_height
            if self.dim_name is None:  # Do it off coordinates, not ID
                if "x_geostationary" == self.x_dim_name:
                    left, bottom, right, top = _convert_to_geostationary(
                        location, xr_data, left, bottom, right, top
                    )

                    x_mask = (left <= xr_data.x_geostationary) & (xr_data.x_geostationary <= right)
                    y_mask = (xr_data.y_geostationary <= top) & (  # Y is flipped
                        bottom <= xr_data.y_geostationary
                    )
                    selected = xr_data.isel(x_geostationary=x_mask, y_geostationary=y_mask)
                elif "longitude" == self.x_dim_name:
                    if location.coordinate_system == "osgb":
                        # Convert to geostationary edges
                        bottom, left = osgb_to_lat_lon(x=left, y=bottom)
                        top, right = osgb_to_lat_lon(x=right, y=top)
                    x_mask = (left <= xr_data.longitude) & (xr_data.longitude <= right)
                    y_mask = (xr_data.latitude <= top) & (  # Y is flipped
                        bottom <= xr_data.latitude
                    )
                    selected = xr_data.isel(longitude=x_mask, latitude=y_mask)
                elif "x" == self.x_dim_name:
                    left, bottom, right, top = _convert_to_geostationary(
                        location, xr_data, left, bottom, right, top
                    )
                    x_mask = (left <= xr_data.x) & (xr_data.x <= right)
                    y_mask = (xr_data.y <= top) & (bottom <= xr_data.y)  # Y is flipped
                    selected = xr_data.isel(x=x_mask, y=y_mask)
                elif "x_osgb" == self.x_dim_name:
                    if location.coordinate_system == "lat_lon":
                        # Convert to OSGB
                        left, bottom = lat_lon_to_osgb(longitude=left, latitude=bottom)
                        right, top = lat_lon_to_osgb(longitude=right, latitude=top)
                    # Select data in the region of interest:
                    x_mask = (left <= xr_data.x_osgb) & (xr_data.x_osgb <= right)
                    y_mask = (xr_data.y_osgb <= top) & (bottom <= xr_data.y_osgb)
                    selected = xr_data.isel(x_osgb=x_mask, y_osgb=y_mask)
                else:
                    raise ValueError(
                        f"{self.x_dim_name=} not in 'x', 'x_osgb',"
                        f" 'x_geostationary', 'longitude', and {self.dim_name=} is 'None'"
                    )
            else:
                # Select data in the region of interest and ID:
                # This also works for unstructured grids
                # Need to check coordinate systems match
                if location.coordinate_system == "osgb" and "longitude" in self.x_dim_name:
                    # Convert to lat_lon edges
                    left, bottom = osgb_to_lat_lon(x=left, y=bottom)
                    right, top = osgb_to_lat_lon(x=right, y=top)
                elif location.coordinate_system == "lat_lon" and "osgb" in self.x_dim_name:
                    left, bottom = lat_lon_to_osgb(longitude=left, latitude=bottom)
                    right, top = lat_lon_to_osgb(longitude=right, latitude=top)
                id_mask = (
                    (left <= getattr(xr_data, self.x_dim_name))
                    & (getattr(xr_data, self.x_dim_name) <= right)
                    & (getattr(xr_data, self.y_dim_name) <= top)
                    & (bottom <= getattr(xr_data, self.y_dim_name))
                )

                selected = xr_data.isel({self.dim_name: id_mask})
            yield selected


def _convert_to_geostationary(location, xr_data, left, bottom, right, top):
    if location.coordinate_system == "osgb":
        # Convert to geostationary edges
        _osgb_to_geostationary = load_geostationary_area_definition_and_transform_osgb(xr_data)
        left, bottom = _osgb_to_geostationary(xx=left, yy=bottom)
        right, top = _osgb_to_geostationary(xx=right, yy=top)
    elif location.coordinate_system == "lat_lon":
        # Convert to geostationary edges
        _lat_lon_to_geostationary = load_geostationary_area_definition_and_transform_latlon(xr_data)
        left, bottom = _lat_lon_to_geostationary(xx=left, yy=bottom)
        right, top = _lat_lon_to_geostationary(xx=right, yy=top)
    return left, bottom, right, top


def _get_idx_of_pixel_closest_to_poi(
    xr_data: xr.DataArray,
    location: Location,
    y_dim_name: str = "y",
    x_dim_name: str = "x",
) -> Location:
    """
    Return x and y index location of pixel at center of region of interest.

    Args:
        xr_data: Xarray dataset
        location: Center in OSGB coordinates
        y_dim_name: Y dimension name
        x_dim_name: X dimension name

    Returns:
        The Location for the center pixel
    """
    y_index = xr_data.get_index(y_dim_name)
    x_index = xr_data.get_index(x_dim_name)
    if location.coordinate_system == "osgb":
        if "x_osgb" == x_dim_name:
            return Location(
                y=y_index.get_indexer([float(location.y)], method="nearest")[0],
                x=x_index.get_indexer([float(location.x)], method="nearest")[0],
                coordinate_system="idx",
            )
        elif "longitude" == x_dim_name:
            latitude, longitude = osgb_to_lat_lon(x=location.x, y=location.y)
            return Location(
                y=y_index.get_indexer([float(latitude)], method="nearest")[0],
                x=x_index.get_indexer([float(longitude)], method="nearest")[0],
                coordinate_system="idx",
            )
        else:
            return NotImplementedError("Only 'x_osgb' and 'longitude' are supported")
    elif location.coordinate_system == "lat_lon":
        if "longitude" == x_dim_name:
            return Location(
                y=y_index.get_indexer([float(location.y)], method="nearest")[0],
                x=x_index.get_indexer([float(location.x)], method="nearest")[0],
                coordinate_system="idx",
            )
        elif "x_osgb" == x_dim_name:
            x_osgb, y_osgb = lat_lon_to_osgb(longitude=location.x, latitude=location.y)
            return Location(
                y=y_index.get_indexer([float(y_osgb)], method="nearest")[0],
                x=x_index.get_indexer([float(x_osgb)], method="nearest")[0],
                coordinate_system="idx",
            )
        else:
            return NotImplementedError("Only 'x_osgb' and 'longitude' are supported")


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
        x=center_geostationary_tuple[0],
        y=center_geostationary_tuple[1],
        coordinate_system="geostationary",
    )

    # Get the index into x and y nearest to x_center_geostationary and y_center_geostationary:
    x_index_at_center = np.searchsorted(xr_data[x_dim_name].values, center_geostationary.x) - 1
    # y_geostationary is in descending order:
    y_index_at_center = len(xr_data[y_dim_name]) - (
        np.searchsorted(xr_data[y_dim_name].values[::-1], center_geostationary.y) - 1
    )
    return Location(x=x_index_at_center, y=y_index_at_center)


def _get_points_from_unstructured_grids(
    xr_data: xr.DataArray,
    location: Location,
    y_dim_name: str = "y",
    x_dim_name: str = "x",
    location_idx_name: str = "values",
    num_points: int = 1,
):
    """
    Get the closest points from an unstructured grid (i.e. Icosahedral grid)

    This is primarily used for the Icosahedral grid, which is not a regular grid,
     and so is not an image

    Args:
        xr_data: Xarray dataset
        location: Location of center point
        y_dim_name: y_dim name
        x_dim_name: x_dim name
        location_idx_name: Name of the index values dimension
            (i.e. where we index into to get the lat/lon for that point)
        num_points: Number of points to return (should be width * height)

    Returns:
        The closest points from the grid
    """
    # Check if need to convert from different coordinate system to lat/lon
    if location.coordinate_system == "osgb":
        latitude, longitude = osgb_to_lat_lon(x=location.x, y=location.y)
        location = Location(
            x=longitude,
            y=latitude,
            coordinate_system="lat_lon",
        )
    elif location.coordinate_system == "geostationary":
        raise NotImplementedError(
            "Does not currently support geostationary coordinates when using unstructured grids"
        )

    # Extract lat, lon, and locidx data
    lat = xr_data[y_dim_name].values
    lon = xr_data[x_dim_name].values
    locidx = xr_data[location_idx_name].values

    # Create a KDTree
    tree = KDTree(list(zip(lat, lon)))

    # Query with the [longitude, latitude] of your point
    _, idx = tree.query([location.x, location.y], k=num_points)

    # Retrieve the location_idxs for these grid points
    location_idxs = locidx[idx]

    data = xr_data.sel({location_idx_name: location_idxs})
    return data
