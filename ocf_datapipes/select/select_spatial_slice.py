"""Select spatial slices"""

import logging
from typing import Optional, Union

import numpy as np
import xarray as xr
from scipy.spatial import KDTree
from torch.utils.data import IterDataPipe, functional_datapipe

from ocf_datapipes.utils import Location
from ocf_datapipes.utils.geospatial import (
    lon_lat_to_geostationary_area_coords,
    lon_lat_to_osgb,
    move_lon_lat_by_meters,
    osgb_to_geostationary_area_coords,
    osgb_to_lon_lat,
    spatial_coord_type,
)
from ocf_datapipes.utils.utils import searchsorted

logger = logging.getLogger(__name__)


# -------------------------------- utility functions --------------------------------


def convert_coords_to_match_xarray(x, y, from_coords, xr_data):
    """Convert x and y coords to cooridnate system matching xarray data

    Args:
        x: Float or array-like
        y: Float or array-like
        from_coords: String describing coordinate system of x and y
        xr_data: xarray data object to which coordinates should be matched
    """

    xr_coords, xr_x_dim, xr_y_dim = spatial_coord_type(xr_data)

    assert from_coords in ["osgb", "lon_lat"]
    assert xr_coords in ["geostationary", "osgb", "lon_lat"]

    if xr_coords == "geostationary":
        if from_coords == "osgb":
            x, y = osgb_to_geostationary_area_coords(x, y, xr_data)

        elif from_coords == "lon_lat":
            x, y = lon_lat_to_geostationary_area_coords(x, y, xr_data)

    elif xr_coords == "lon_lat":
        if from_coords == "osgb":
            x, y = osgb_to_lon_lat(x, y)

        # else the from_coords=="lon_lat" and we don't need to convert

    elif xr_coords == "osgb":
        if from_coords == "lon_lat":
            x, y = lon_lat_to_osgb(x, y)

        # else the from_coords=="osgb" and we don't need to convert

    return x, y


def _get_idx_of_pixel_closest_to_poi(
    xr_data: xr.DataArray,
    location: Location,
) -> Location:
    """
    Return x and y index location of pixel at center of region of interest.

    Args:
        xr_data: Xarray dataset
        location: Center
    Returns:
        The Location for the center pixel
    """
    xr_coords, xr_x_dim, xr_y_dim = spatial_coord_type(xr_data)

    if xr_coords not in ["osgb", "lon_lat"]:
        raise NotImplementedError(f"Only 'osgb' and 'lon_lat' are supported - not '{xr_coords}'")

    # Convert location coords to match xarray data
    x, y = convert_coords_to_match_xarray(
        location.x,
        location.y,
        from_coords=location.coordinate_system,
        xr_data=xr_data,
    )

    # Check that the requested point lies within the data
    assert xr_data[xr_x_dim].min() < x < xr_data[xr_x_dim].max()
    assert xr_data[xr_y_dim].min() < y < xr_data[xr_y_dim].max()

    x_index = xr_data.get_index(xr_x_dim)
    y_index = xr_data.get_index(xr_y_dim)

    closest_x = x_index.get_indexer([x], method="nearest")[0]
    closest_y = y_index.get_indexer([y], method="nearest")[0]

    return Location(x=closest_x, y=closest_y, coordinate_system="idx")


def _get_idx_of_pixel_closest_to_poi_geostationary(
    xr_data: xr.DataArray,
    center_coordinate: Location,
) -> Location:
    """
    Return x and y index location of pixel at center of region of interest.

    Args:
        xr_data: Xarray dataset
        center_coordinate: Central coordinate

    Returns:
        Location for the center pixel in geostationary coordinates
    """
    xr_coords, xr_x_dim, xr_y_dim = spatial_coord_type(xr_data)
    if center_coordinate.coordinate_system == "osgb":
        x, y = osgb_to_geostationary_area_coords(
            x=center_coordinate.x, y=center_coordinate.y, xr_data=xr_data
        )
    elif center_coordinate.coordinate_system == "lon_lat":
        x, y = lon_lat_to_geostationary_area_coords(
            x=center_coordinate.x, y=center_coordinate.y, xr_data=xr_data
        )
    else:
        raise NotImplementedError(
            f"Only 'osgb' and 'lon_lat' location coordinates are \
                                  supported in conversion to geostationary \
                                   - not '{center_coordinate.coordinate_system}'"
        )

    center_geostationary = Location(x=x, y=y, coordinate_system="geostationary")
    # Check that the requested point lies within the data
    assert xr_data[xr_x_dim].min() < x < xr_data[xr_x_dim].max()
    assert xr_data[xr_y_dim].min() < y < xr_data[xr_y_dim].max()

    # Get the index into x and y nearest to x_center_geostationary and y_center_geostationary:
    x_index_at_center = searchsorted(
        xr_data[xr_x_dim].values, center_geostationary.x, assume_ascending=True
    )

    # y_geostationary is in descending order:
    y_index_at_center = searchsorted(
        xr_data[xr_y_dim].values, center_geostationary.y, assume_ascending=False
    )

    return Location(x=x_index_at_center, y=y_index_at_center, coordinate_system="idx")


def _get_points_from_unstructured_grids(
    xr_data: xr.DataArray,
    location: Location,
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
        location_idx_name: Name of the index values dimension
            (i.e. where we index into to get the lat/lon for that point)
        num_points: Number of points to return (should be width * height)

    Returns:
        The closest points from the grid
    """
    xr_coords, xr_x_dim, xr_y_dim = spatial_coord_type(xr_data)
    assert xr_coords == "lon_lat"

    # Check if need to convert from different coordinate system to lat/lon
    if location.coordinate_system == "osgb":
        longitude, latitude = osgb_to_lon_lat(x=location.x, y=location.y)
        location = Location(
            x=longitude,
            y=latitude,
            coordinate_system="lon_lat",
        )
    elif location.coordinate_system == "geostationary":
        raise NotImplementedError(
            "Does not currently support geostationary coordinates when using unstructured grids"
        )

    # Extract lat, lon, and locidx data
    lat = xr_data.longitude.values
    lon = xr_data.latitude.values
    locidx = xr_data[location_idx_name].values

    # Create a KDTree
    tree = KDTree(list(zip(lat, lon)))

    # Query with the [longitude, latitude] of your point
    _, idx = tree.query([location.x, location.y], k=num_points)

    # Retrieve the location_idxs for these grid points
    location_idxs = locidx[idx]

    data = xr_data.sel({location_idx_name: location_idxs})
    return data


# ---------------------------- sub-functions for slicing ----------------------------


def _slice_patial_spatial_pixel_window_from_xarray(
    xr_data,
    left_idx,
    right_idx,
    top_idx,
    bottom_idx,
    left_pad_pixels,
    right_pad_pixels,
    top_pad_pixels,
    bottom_pad_pixels,
    xr_x_dim,
    xr_y_dim,
):
    """Return spatial window of given pixel size when window partially overlaps input data"""

    dx = np.median(np.diff(xr_data[xr_x_dim].values))
    dy = np.median(np.diff(xr_data[xr_y_dim].values))

    if left_pad_pixels > 0:
        assert right_pad_pixels == 0
        x_sel = np.concatenate(
            [
                xr_data[xr_x_dim].values[0] - np.arange(left_pad_pixels, 0, -1) * dx,
                xr_data[xr_x_dim].values[0:right_idx],
            ]
        )
        xr_data = xr_data.isel({xr_x_dim: slice(0, right_idx)}).reindex({xr_x_dim: x_sel})

    elif right_pad_pixels > 0:
        assert left_pad_pixels == 0
        x_sel = np.concatenate(
            [
                xr_data[xr_x_dim].values[left_idx:],
                xr_data[xr_x_dim].values[-1] + np.arange(1, right_pad_pixels + 1) * dx,
            ]
        )
        xr_data = xr_data.isel({xr_x_dim: slice(left_idx, None)}).reindex({xr_x_dim: x_sel})

    else:
        xr_data = xr_data.isel({xr_x_dim: slice(left_idx, right_idx)})

    if top_pad_pixels > 0:
        assert bottom_pad_pixels == 0
        y_sel = np.concatenate(
            [
                xr_data[xr_y_dim].values[0] - np.arange(top_pad_pixels, 0, -1) * dy,
                xr_data[xr_y_dim].values[0:bottom_idx],
            ]
        )
        xr_data = xr_data.isel({xr_y_dim: slice(0, bottom_idx)}).reindex({xr_y_dim: y_sel})

    elif bottom_pad_pixels > 0:
        assert top_pad_pixels == 0
        y_sel = np.concatenate(
            [
                xr_data[xr_y_dim].values[top_idx:],
                xr_data[xr_y_dim].values[-1] + np.arange(1, bottom_pad_pixels + 1) * dy,
            ]
        )
        xr_data = xr_data.isel({xr_y_dim: slice(top_idx, None)}).reindex({xr_x_dim: y_sel})

    else:
        xr_data = xr_data.isel({xr_y_dim: slice(top_idx, bottom_idx)})

    return xr_data


def slice_spatial_pixel_window_from_xarray(
    xr_data, center_idx, width_pixels, height_pixels, xr_x_dim, xr_y_dim, allow_partial_slice
):
    """Select a spatial slice from an xarray object

    Args:
        xr_data: Xarray object
        center_idx: Location object describing the centre of the window
        width_pixels: Window with in pixels
        height_pixels: Window height in pixels
        xr_x_dim: Name of the x-dimension in the xr_data
        xr_y_dim: Name of the y-dimension in the xr_data
        allow_partial_slice: Whether to allow a partially filled window
    """
    half_width = width_pixels // 2
    half_height = height_pixels // 2

    left_idx = int(center_idx.x - half_width)
    right_idx = int(center_idx.x + half_width)
    top_idx = int(center_idx.y - half_height)
    bottom_idx = int(center_idx.y + half_height)

    data_width_pixels = len(xr_data[xr_x_dim])
    data_height_pixels = len(xr_data[xr_y_dim])

    left_pad_required = left_idx < 0
    right_pad_required = right_idx >= data_width_pixels
    top_pad_required = top_idx < 0
    bottom_pad_required = bottom_idx >= data_height_pixels

    pad_required = any(
        [left_pad_required, right_pad_required, top_pad_required, bottom_pad_required]
    )

    if pad_required:
        if allow_partial_slice:
            left_pad_pixels = (-left_idx) if left_pad_required else 0
            right_pad_pixels = (right_idx - (data_width_pixels - 1)) if right_pad_required else 0
            top_pad_pixels = (-top_idx) if top_pad_required else 0
            bottom_pad_pixels = (
                (bottom_idx - (data_height_pixels - 1)) if bottom_pad_required else 0
            )

            xr_data = _slice_patial_spatial_pixel_window_from_xarray(
                xr_data,
                left_idx,
                right_idx,
                top_idx,
                bottom_idx,
                left_pad_pixels,
                right_pad_pixels,
                top_pad_pixels,
                bottom_pad_pixels,
                xr_x_dim,
                xr_y_dim,
            )
        else:
            raise ValueError(
                f"Window for location {center_idx} not available. Missing (left, right, top, "
                f"bottom) pixels  = ({left_pad_required}, {right_pad_required}, "
                f"{top_pad_required}, {bottom_pad_required}). "
                f"You may wish to set `allow_partial_slice=True`"
            )

    else:
        xr_data = xr_data.isel(
            {
                xr_x_dim: slice(left_idx, right_idx),
                xr_y_dim: slice(top_idx, bottom_idx),
            }
        )

    assert len(xr_data[xr_x_dim]) == width_pixels, (
        f"Expected x-dim len {width_pixels} got {len(xr_data[xr_x_dim])} "
        f"for location {center_idx} for slice {left_idx}:{right_idx}"
    )
    assert len(xr_data[xr_y_dim]) == height_pixels, (
        f"Expected y-dim len {height_pixels} got {len(xr_data[xr_y_dim])} "
        f"for location {center_idx} for slice {top_idx}:{bottom_idx}"
    )

    return xr_data


# ---------------------------- main functions for slicing ---------------------------


def select_spatial_slice_pixels(
    xr_data: Union[xr.Dataset, xr.DataArray],
    location: Location,
    roi_width_pixels: int,
    roi_height_pixels: int,
    allow_partial_slice: bool = False,
    location_idx_name: Optional[str] = None,
):
    """
    Select spatial slice based off pixels from location point of interest

    If `allow_partial_slice` is set to True, then slices may be made which intersect the border
    of the input data. The additional x and y cordinates that would be required for this slice
    are extrapolated based on the average spacing of these coordinates in the input data.
    However, currently slices cannot be made where the centre of the window is outside of the
    input data.

    Args:
        xr_data: Xarray DataArray or Dataset to slice from
        location: Location of interest
        roi_height_pixels: ROI height in pixels
        roi_width_pixels: ROI width in pixels
        allow_partial_slice: Whether to allow a partial slice.
        location_idx_name: Name for location index of unstructured grid data,
            None if not relevant
    """

    xr_coords, xr_x_dim, xr_y_dim = spatial_coord_type(xr_data)
    if location_idx_name is not None:
        selected = _get_points_from_unstructured_grids(
            xr_data=xr_data,
            location=location,
            location_idx_name=location_idx_name,
            num_points=roi_width_pixels * roi_height_pixels,
        )
    else:
        if xr_coords == "geostationary":
            center_idx: Location = _get_idx_of_pixel_closest_to_poi_geostationary(
                xr_data=xr_data,
                center_coordinate=location,
            )
        else:
            center_idx: Location = _get_idx_of_pixel_closest_to_poi(
                xr_data=xr_data,
                location=location,
            )

        selected = slice_spatial_pixel_window_from_xarray(
            xr_data,
            center_idx,
            roi_width_pixels,
            roi_height_pixels,
            xr_x_dim,
            xr_y_dim,
            allow_partial_slice=allow_partial_slice,
        )

    return selected


def select_spatial_slice_meters(
    xr_data: Union[xr.Dataset, xr.DataArray],
    location: Location,
    roi_width_meters: int,
    roi_height_meters: int,
    dim_name: Optional[str] = None,
):
    """
    Select spatial slice based off pixels from point of interest

    Args:
        xr_data: Xarray DataArray or Dataset to slice from
        location: Location of interest
        roi_height_meters: ROI height in meters
        roi_width_meters: ROI width in meters
        dim_name: Dimension name to select for ID, None for coordinates

    Notes:
        Using spatial slicing based on distance rather than number of pixels will often yield
        slices which can vary by 1 pixel in height and/or width.

        E.g. Suppose the Xarray data has x-coords = [1,2,3,4,5]. We want to slice a spatial
        window with a size which equates to 2.2 along the x-axis. If we choose to slice around
        the point x=3 this will slice out the x-coords [2,3,4]. If we choose to slice around the
        point x=2.5 this will slice out the x-coords [2,3]. Hence the returned slice can have
        size either 2 or 3 in the x-axis depending on the spatial location selected.

        Also, if selecting over a large span of latitudes, this may also causes pixel sizes of
        the yielded outputs to change. For example, if the Xarray data is on a regularly spaced
        longitude-latitude grid, then the structure of the grid means that the longitudes near
        to the poles are spaced closer together (measured in meters) than at the equator. So
        slices near the equator will have less pixels in the x-axis than slices taken near the
        poles.
    """
    # Get the spatial coords of the xarray data
    xr_coords, xr_x_dim, xr_y_dim = spatial_coord_type(xr_data)

    half_width = roi_width_meters // 2
    half_height = roi_height_meters // 2

    # Find the bounding box values for the location in either lat-lon or OSGB coord systems
    if location.coordinate_system == "lon_lat":
        right, top = move_lon_lat_by_meters(
            location.x,
            location.y,
            half_width,
            half_height,
        )
        left, bottom = move_lon_lat_by_meters(
            location.x,
            location.y,
            -half_width,
            -half_height,
        )

    elif location.coordinate_system == "osgb":
        left = location.x - half_width
        right = location.x + half_width
        bottom = location.y - half_height
        top = location.y + half_height

    else:
        raise ValueError(f"Location coord system not recognized: {location.coordinate_system}")

    # Change the bounding coordinates [left, right, bottom, top] to the same
    # coordinate system as the xarray data
    (left, right), (bottom, top) = convert_coords_to_match_xarray(
        x=np.array([left, right], dtype=np.float32),
        y=np.array([bottom, top], dtype=np.float32),
        from_coords=location.coordinate_system,
        xr_data=xr_data,
    )

    # Do it off coordinates, not ID
    if dim_name is None:
        # Select a patch from the xarray data
        x_mask = (left <= xr_data[xr_x_dim]) & (xr_data[xr_x_dim] <= right)
        y_mask = (bottom <= xr_data[xr_y_dim]) & (xr_data[xr_y_dim] <= top)
        selected = xr_data.isel({xr_x_dim: x_mask, xr_y_dim: y_mask})

    else:
        # Select data in the region of interest and ID:
        # This also works for unstructured grids

        id_mask = (
            (left <= xr_data[xr_x_dim])
            & (xr_data[xr_x_dim] <= right)
            & (bottom <= xr_data[xr_y_dim])
            & (xr_data[xr_y_dim] <= top)
        )
        selected = xr_data.isel({dim_name: id_mask})
    return selected


# ------------------------------ datapipes for slicing ------------------------------


@functional_datapipe("select_spatial_slice_pixels")
class SelectSpatialSlicePixelsIterDataPipe(IterDataPipe):
    """Select spatial slice based off pixels from point of interest"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        location_datapipe: IterDataPipe,
        roi_width_pixels: int,
        roi_height_pixels: int,
        allow_partial_slice: bool = False,
        location_idx_name: Optional[str] = None,
    ):
        """
        Select spatial slice based off pixels from point of interest

        If `allow_partial_slice` is set to True, then slices may be made which intersect the border
        of the input data. The additional x and y cordinates that would be required for this slice
        are extrapolated based on the average spacing of these coordinates in the input data.
        However, currently slices cannot be made where the centre of the window is outside of the
        input data.

        Args:
            source_datapipe: Datapipe of Xarray data
            location_datapipe: Location datapipe
            roi_height_pixels: ROI height in pixels
            roi_width_pixels: ROI width in pixels
            allow_partial_slice: Whether to allow a partial slice.
            location_idx_name: Name for location index of unstructured grid data,
                None if not relevant
        """
        self.source_datapipe = source_datapipe
        self.location_datapipe = location_datapipe
        self.roi_height_pixels = roi_height_pixels
        self.roi_width_pixels = roi_width_pixels
        self.allow_partial_slice = allow_partial_slice
        self.location_idx_name = location_idx_name

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        for xr_data, location in self.source_datapipe.zip_ocf(self.location_datapipe):
            logger.debug("Selecting spatial slice with pixels")

            selected = select_spatial_slice_pixels(
                xr_data=xr_data,
                location=location,
                roi_width_pixels=self.roi_width_pixels,
                roi_height_pixels=self.roi_height_pixels,
                allow_partial_slice=self.allow_partial_slice,
                location_idx_name=self.location_idx_name,
            )

            yield selected


@functional_datapipe("select_spatial_slice_meters")
class SelectSpatialSliceMetersIterDataPipe(IterDataPipe):
    """Select spatial slice based off meters from point of interest"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        location_datapipe: IterDataPipe,
        roi_width_meters: int,
        roi_height_meters: int,
        dim_name: Optional[str] = None,
    ):
        """
        Select spatial slice based off pixels from point of interest

        Args:
            source_datapipe: Datapipe of Xarray data
            location_datapipe: Location datapipe
            roi_width_meters: ROI width in meters
            roi_height_meters: ROI height in meters
            dim_name: Dimension name to select for ID, None for coordinates

        Notes:
            Using spatial slicing based on distance rather than number of pixels will often yield
            slices which can vary by 1 pixel in height and/or width.

            E.g. Suppose the Xarray data has x-coords = [1,2,3,4,5]. We want to slice a spatial
            window with a size which equates to 2.2 along the x-axis. If we choose to slice around
            the point x=3 this will slice out the x-coords [2,3,4]. If we choose to slice around the
            point x=2.5 this will slice out the x-coords [2,3]. Hence the returned slice can have
            size either 2 or 3 in the x-axis depending on the spatial location selected.

            Also, if selecting over a large span of latitudes, this may also causes pixel sizes of
            the yielded outputs to change. For example, if the Xarray data is on a regularly spaced
            longitude-latitude grid, then the structure of the grid means that the longitudes near
            to the poles are spaced closer together (measured in meters) than at the equator. So
            slices near the equator will have less pixels in the x-axis than slices taken near the
            poles.
        """
        self.source_datapipe = source_datapipe
        self.location_datapipe = location_datapipe
        self.roi_height_meters = roi_height_meters
        self.roi_width_meters = roi_width_meters
        self.dim_name = dim_name

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        for xr_data, location in self.source_datapipe.zip_ocf(self.location_datapipe):
            # Compute the index for left and right:
            logger.debug("Getting Spatial Slice Meters")

            selected = select_spatial_slice_meters(
                xr_data=xr_data,
                location=location,
                roi_width_meters=self.roi_width_meters,
                roi_height_meters=self.roi_height_meters,
                dim_name=self.dim_name,
            )

            yield selected
