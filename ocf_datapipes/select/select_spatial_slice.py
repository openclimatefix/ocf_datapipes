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
    spatial_coord_type,
)
from ocf_datapipes.utils.utils import searchsorted

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
        location_idx_name: Optional[str] = None,
    ):
        """
        Select spatial slice based off pixels from point of interest

        Args:
            source_datapipe: Datapipe of Xarray data
            location_datapipe: Location datapipe
            roi_height_pixels: ROI height in pixels
            roi_width_pixels: ROI width in pixels
            location_idx_name: Name for location index of unstructured grid data,
                None if not relevant
        """
        self.source_datapipe = source_datapipe
        self.location_datapipe = location_datapipe
        self.roi_height_pixels = roi_height_pixels
        self.roi_width_pixels = roi_width_pixels
        self.location_idx_name = location_idx_name

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        for xr_data, location in self.source_datapipe.zip_ocf(self.location_datapipe):
            logger.debug("Selecting spatial slice with pixels")
            
            xr_coords, xr_x_dim, xr_y_dim = spatial_coord_type(xr_data)
            
            if self.location_idx_name is not None:
                selected = _get_points_from_unstructured_grids(
                    xr_data=xr_data,
                    location=location,
                    location_idx_name=self.location_idx_name,
                    num_points=self.roi_width_pixels * self.roi_height_pixels,
                )
                yield selected

            if xr_coords=="geostationary":
                center_idx: Location = _get_idx_of_pixel_closest_to_poi_geostationary(
                    xr_data=xr_data,
                    center_osgb=location,
                )
            else:
                center_idx: Location = _get_idx_of_pixel_closest_to_poi(
                    xr_data=xr_data,
                    location=location,
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
            data_width_pixels = len(xr_data[xr_x_dim])
            assert right_idx <= data_width_pixels, f"{right_idx=} must be <= {data_width_pixels=}"
            assert top_idx >= 0, f"{top_idx=} must be >= 0!"
            data_height_pixels = len(xr_data[xr_y_dim])
            assert (
                bottom_idx <= data_height_pixels
            ), f"{bottom_idx=} must be <= {data_height_pixels=}"

            selected = xr_data.isel(
                {
                    xr_x_dim: slice(left_idx, right_idx),
                    xr_y_dim: slice(top_idx, bottom_idx),
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
        dim_name: Optional[str] = None, #"pv_system_id",
    ):
        """
        Select spatial slice based off pixels from point of interest

        Args:
            source_datapipe: Datapipe of Xarray data
            location_datapipe: Location datapipe
            roi_height_meters: ROI height in meters
            roi_width_meters: ROI width in meters
            dim_name: Dimension name to select for ID, None for coordinates
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

            xr_coords, xr_x_dim, xr_y_dim = spatial_coord_type(xr_data)
            
            half_height = self.roi_height_meters // 2
            half_width = self.roi_width_meters // 2
            
            # Find the bounding box values for the location in either lat-lon or OSGB coord systems
            if location.coordinate_system == "lat_lon":
                top, right = move_lat_lon_by_meters(
                    location.y, location.x, half_height, half_width
                )
                bottom, left = move_lat_lon_by_meters(
                    location.y, location.x, -half_height, -half_width
                )
                
            elif location.coordinate_system == "osgb":
                left = location.x - half_width
                right = location.x + half_width
                bottom = location.y - half_height
                top = location.y + half_height
                
            else:
                raise ValueError(
                    f"Location coord system not recognized: {location.coordinate_system}")
            
            if self.dim_name is None:  # Do it off coordinates, not ID
                
                if xr_coords=="geostationary":
                    left, bottom, right, top = _convert_to_geostationary(
                        location, xr_data, left, bottom, right, top
                    )
                    
                    x_mask = (left <= xr_data.x_geostationary) & (xr_data.x_geostationary <= right)
                    y_mask = (bottom <= xr_data.y_geostationary) & (xr_data.y_geostationary <= top)
                    
                    selected = xr_data.isel(x_geostationary=x_mask, y_geostationary=y_mask)
                
                elif xr_coords=="lat_lon":
                    
                    if location.coordinate_system == "osgb":
                        bottom, left = osgb_to_lat_lon(x=left, y=bottom)
                        top, right = osgb_to_lat_lon(x=right, y=top)
                    
                    x_mask = (left <= xr_data.longitude) & (xr_data.longitude <= right)
                    y_mask =  (bottom <= xr_data.latitude) & (xr_data.latitude <= top)
                    selected = xr_data.isel(longitude=x_mask, latitude=y_mask)
                
                elif xr_coords=="osgb":
                    
                    if location.coordinate_system == "lat_lon":
                        left, bottom = lat_lon_to_osgb(longitude=left, latitude=bottom)
                        right, top = lat_lon_to_osgb(longitude=right, latitude=top)
                        
                    x_mask = (left <= xr_data.longitude) & (xr_data.longitude <= right)
                    y_mask = (bottom <= xr_data.latitude) & (xr_data.latitude <= top)
                    
                    x_mask = (left <= xr_data.x_osgb) & (xr_data.x_osgb <= right)
                    y_mask = (bottom <= xr_data.y_osgb) & (xr_data.y_osgb <= top)
                    selected = xr_data.isel(x_osgb=x_mask, y_osgb=y_mask)

            else:
                # Select data in the region of interest and ID:
                # This also works for unstructured grids
                # Need to check coordinate systems match
                if location.coordinate_system == "osgb" and xr_coords=="lat_lon":
                    # Convert to lat_lon edges
                    left, bottom = osgb_to_lat_lon(x=left, y=bottom)
                    right, top = osgb_to_lat_lon(x=right, y=top)

                elif location.coordinate_system == "lat_lon" and xr_coords=="osgb":
                    left, bottom = lat_lon_to_osgb(longitude=left, latitude=bottom)
                    right, top = lat_lon_to_osgb(longitude=right, latitude=top)
                    
                id_mask = (
                    (left <= xr_data[xr_x_dim]) & (xr_data[xr_x_dim] <= right) &
                    (bottom <= xr_data[xr_y_dim]) & (xr_data[xr_y_dim] <= top)
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
    
    y_index = xr_data.get_index(xr_y_dim)
    x_index = xr_data.get_index(xr_x_dim)
    
    if location.coordinate_system == "osgb":
        if xr_coords=="osgb":
            return Location(
                y=y_index.get_indexer([float(location.y)], method="nearest")[0],
                x=x_index.get_indexer([float(location.x)], method="nearest")[0],
                coordinate_system="idx",
            )
        elif xr_coords=="lat_lon":
            latitude, longitude = osgb_to_lat_lon(x=location.x, y=location.y)
            return Location(
                y=y_index.get_indexer([float(latitude)], method="nearest")[0],
                x=x_index.get_indexer([float(longitude)], method="nearest")[0],
                coordinate_system="idx",
            )
        else:
            raise NotImplementedError(
                f"Only 'osgb' and 'lat_lon' are supported - not '{xr_coords}'"
            )
    
    elif location.coordinate_system == "lat_lon":
        if xr_coords=="lat_lon":
            return Location(
                y=y_index.get_indexer([float(location.y)], method="nearest")[0],
                x=x_index.get_indexer([float(location.x)], method="nearest")[0],
                coordinate_system="idx",
            )
        elif xr_coords=="osgb":
            x_osgb, y_osgb = lat_lon_to_osgb(longitude=location.x, latitude=location.y)
            return Location(
                y=y_index.get_indexer([float(y_osgb)], method="nearest")[0],
                x=x_index.get_indexer([float(x_osgb)], method="nearest")[0],
                coordinate_system="idx",
            )
        else:
            raise NotImplementedError(
                f"Only 'osgb' and 'lat_lon' are supported - not '{xr_coords}'"
            )


def _get_idx_of_pixel_closest_to_poi_geostationary(
    xr_data: xr.DataArray,
    center_osgb: Location,
) -> Location:
    """
    Return x and y index location of pixel at center of region of interest.

    Args:
        xr_data: Xarray dataset
        center_osgb: Center in OSGB coordinates

    Returns:
        Location for the center pixel in geostationary coordinates
    """    
    
    xr_coords, xr_x_dim, xr_y_dim = spatial_coord_type(xr_data)
    
    _osgb_to_geostationary = load_geostationary_area_definition_and_transform_osgb(xr_data)
    x, y = _osgb_to_geostationary(xx=center_osgb.x, yy=center_osgb.y)
    center_geostationary = Location(x=x, y=y, coordinate_system="geostationary")

    # Get the index into x and y nearest to x_center_geostationary and y_center_geostationary:
    x_index_at_center = searchsorted(
        xr_data[xr_x_dim].values, 
        center_geostationary.x,
        assume_ascending=True
    )

    # y_geostationary is in descending order:
    y_index_at_center = searchsorted(
        xr_data[xr_y_dim].values, 
        center_geostationary.y,
        assume_ascending=False
    )
    
    return Location(x=x_index_at_center, y=y_index_at_center)


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
    assert xr_coords=="lat_lon"
    
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
