from torchdata.datapipes.iter import IterDataPipe, Zipper
from torchdata.datapipes import functional_datapipe
import xarray as xr
from typing import Union
from ocf_datapipes.utils.consts import Location

@functional_datapipe("select_spatial_slice_pixels")
class SelectSpatialSlicePixelsIterDataPipe(IterDataPipe):
    """Select spatial slice based off pixels from point of interest"""
    def __init__(self, source_datapipe: IterDataPipe, location_datapipe: IterDataPipe, roi_height_pixels: int, roi_width_pixels: int):
        self.source_datapipe = source_datapipe
        self.location_datapipe = location_datapipe
        self.roi_height_pixels = roi_height_pixels
        self.roi_width_pixels = roi_width_pixels

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        for xr_data, location in Zipper(self.source_datapipe, self.location_datapipe):
            center_idx = _get_idx_of_pixel_closest_to_poi(
                xr_data=xr_data, center_osgb=location
            )

            # Compute the index for left and right:
            half_height = self.roi_height_pixels // 2
            half_width = self.roi_width_pixels // 2

            left_idx = center_idx.x - half_width
            right_idx = center_idx.x + half_width
            top_idx = center_idx.y - half_height
            bottom_idx = center_idx.y + half_height

            # Sanity check!
            assert left_idx >= 0, f"{left_idx=} must be >= 0!"
            data_width_pixels = len(xr_data["x"])
            assert right_idx <= data_width_pixels, f"{right_idx=} must be <= {data_width_pixels=}"
            assert top_idx >= 0, f"{top_idx=} must be >= 0!"
            data_height_pixels = len(xr_data["y"])
            assert bottom_idx <= data_height_pixels, f"{bottom_idx=} must be <= {data_height_pixels=}"

            selected = xr_data.isel(
                {
                    "x": slice(left_idx, right_idx),
                    "y": slice(top_idx, bottom_idx),
                }
            )
            yield selected


@functional_datapipe("select_spatial_slice_meters")
class SelectSpatialSliceMetersIterDataPipe(IterDataPipe):
    """Select spatial slice based off meters from point of interest"""
    def __init__(self, source_datapipe: IterDataPipe, location_datapipe: IterDataPipe, roi_height_meters: int, roi_width_meters: int):
        self.source_datapipe = source_datapipe
        self.location_datapipe = location_datapipe
        self.roi_height_meters = roi_height_meters
        self.roi_width_meters = roi_width_meters

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        for xr_data, location in Zipper(self.source_datapipe, self.location_datapipe):
            # Compute the index for left and right:
            half_height = self.roi_height_meters // 2
            half_width = self.roi_width_meters // 2

            left_idx = location.x - half_width
            right_idx = location.x + half_width
            top_idx = location.y - half_height
            bottom_idx = location.y + half_height

            selected = xr_data.isel(
                {
                    "x": slice(left_idx, right_idx),
                    "y": slice(top_idx, bottom_idx),
                }
            )
            yield selected

def _get_idx_of_pixel_closest_to_poi(
    xr_data: xr.DataArray, center_osgb: Location
) -> Location:
    """Return x and y index location of pixel at center of region of interest."""
    y_index = xr_data.get_index("y")
    x_index = xr_data.get_index("x")
    return Location(
        y=y_index.get_indexer([center_osgb.y], method="nearest")[0],
        x=x_index.get_indexer([center_osgb.x], method="nearest")[0],
    )