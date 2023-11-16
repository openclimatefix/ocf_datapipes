"""Preprocessing for MetNet-type inputs"""
import itertools
from typing import List

import numpy as np
import pvlib
import xarray as xr
from torch.utils.data import IterDataPipe, functional_datapipe

from ocf_datapipes.select.select_spatial_slice import convert_coords_to_match_xarray
from ocf_datapipes.utils import Zipper
from ocf_datapipes.utils.geospatial import (
    geostationary_area_coords_to_lonlat,
    move_lon_lat_by_meters,
    osgb_to_lon_lat,
    spatial_coord_type,
)
from ocf_datapipes.utils.parallel import run_with_threadpool
from ocf_datapipes.utils.utils import trigonometric_datetime_transformation

ELEVATION_MEAN = 37.4
ELEVATION_STD = 12.7
AZIMUTH_MEAN = 177.7
AZIMUTH_STD = 41.7


@functional_datapipe("preprocess_metnet")
class PreProcessMetNetIterDataPipe(IterDataPipe):
    """Preprocess set of Xarray datasets similar to MetNet-1"""

    def __init__(
        self,
        source_datapipes: List[IterDataPipe],
        location_datapipe: IterDataPipe,
        context_width: int,
        context_height: int,
        center_width: int,
        center_height: int,
        output_height_pixels: int,
        output_width_pixels: int,
        add_sun_features: bool = False,
        only_sun: bool = False,
    ):
        """

        Processes set of Xarray datasets similar to MetNet

        In terms of taking all available source datapipes:
        1. selecting the same context area of interest
        2. Creating a center crop of the center_height, center_width
        3. Downsampling the context area of interest to the same shape as the center crop
        4. Stacking those context images on the center crop.
        5. Add Month, Day, Hour channels for each input time
        6. Add Sun position as well?

        This would be designed originally for NWP+Satellite+Topographic data sources.
        To add the PV power for lots of sites, the PV power would
        need to be able to be on a grid for the context/center
        crops and then for the downsample

        This also appends Lat/Lon coordinates to the stack,
         and returns a new Numpy array with the stacked data

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
            add_sun_features: Whether to calculate and
            add Sun elevation and azimuth for each center pixel
            only_sun: Whether to only output sun features
                Assumes only one input to give the coordinates
        """
        self.source_datapipes = source_datapipes
        self.location_datapipe = location_datapipe
        self.context_width = context_width
        self.context_height = context_height
        self.center_width = center_width
        self.center_height = center_height
        self.output_height_pixels = output_height_pixels
        self.output_width_pixels = output_width_pixels
        self.add_sun_features = add_sun_features
        self.only_sun = only_sun

    def __iter__(self) -> np.ndarray:
        for xr_datas, location in Zipper(Zipper(*self.source_datapipes), self.location_datapipe):
            # TODO Use the Lat/Long coordinates of the center array for the lat/lon stuff
            # Do the resampling and cropping in parallel
            xr_datas = run_with_threadpool(
                zip(
                    _bicycle(xr_datas),
                    itertools.repeat(location),
                    itertools.chain.from_iterable(
                        zip(
                            itertools.repeat(self.center_width),
                            itertools.repeat(self.context_width),
                        )
                    ),
                    itertools.chain.from_iterable(
                        zip(
                            itertools.repeat(self.center_height),
                            itertools.repeat(self.context_height),
                        )
                    ),
                    itertools.repeat(self.output_height_pixels),
                    itertools.repeat(self.output_width_pixels),
                ),
                _crop_and_resample_wrapper,
                max_workers=8,
                scheduled_tasks=int(len(xr_datas) * 2),  # One for center, one for context
            )
            xr_datas = list(xr_datas)
            # Output is then list of center, context, center, context, etc.
            # So we need to split the list into two lists of the same length,
            # one with centers, one with contexts
            centers = xr_datas[::2]
            contexts = xr_datas[1::2]
            # Now do the first one for the sun and other features
            xr_center = centers[0]
            _extra_time_dim = (
                "target_time_utc" if "target_time_utc" in xr_center.dims else "time_utc"
            )
            # Add in time features for each timestep
            time_image = _create_time_image(
                xr_center,
                time_dim=_extra_time_dim,
                output_height_pixels=self.output_height_pixels,
                output_width_pixels=self.output_width_pixels,
            )
            contexts.append(time_image)
            # Need to add sun features
            if self.add_sun_features:
                sun_image = _create_sun_image(
                    image_xr=xr_center,
                    x_dim="x_osgb" if "x_osgb" in xr_center.dims else "x_geostationary",
                    y_dim="y_osgb" if "y_osgb" in xr_center.dims else "y_geostationary",
                    time_dim=_extra_time_dim,
                    normalize=True,
                )
                if self.only_sun:
                    contexts = [time_image, sun_image]
                else:
                    contexts.append(sun_image)
            for xr_index in range(len(centers)):
                xr_center = centers[xr_index]
                xr_context = contexts[xr_index]
                xr_center = xr_center.to_numpy()
                xr_context = xr_context.to_numpy()
                if len(xr_center.shape) == 2:  # Need to add channel dimension
                    xr_center = np.expand_dims(xr_center, axis=0)
                    xr_context = np.expand_dims(xr_context, axis=0)
                if len(xr_center.shape) == 3:  # Need to add channel dimension
                    xr_center = np.expand_dims(xr_center, axis=1)
                    xr_context = np.expand_dims(xr_context, axis=1)
                centers[xr_index] = xr_center
                contexts[xr_index] = xr_context
            # Pad out time dimension to be the same, using the largest one
            # All should have 4 dimensions at this point
            max_time_len = max(
                np.max([c.shape[0] for c in centers]), np.max([c.shape[0] for c in contexts])
            )
            for i in range(len(centers)):
                centers[i] = np.pad(
                    centers[i],
                    pad_width=(
                        (0, max_time_len - centers[i].shape[0]),
                        (0, 0),
                        (0, 0),
                        (0, 0),
                    ),
                    mode="constant",
                    constant_values=0.0,
                )
            for i in range(len(contexts)):
                contexts[i] = np.pad(
                    contexts[i],
                    pad_width=(
                        (0, max_time_len - contexts[i].shape[0]),
                        (0, 0),
                        (0, 0),
                        (0, 0),
                    ),
                    mode="constant",
                    constant_values=0.0,
                )
            stacked_data = np.concatenate([*centers, *contexts], axis=1)
            yield stacked_data


def _crop_and_resample_wrapper(args):
    return _crop_and_resample(*args)


def _bicycle(xr_datas):
    for xr_data in xr_datas:
        yield xr_data
        yield xr_data


def _crop_and_resample(
    xr_data: xr.Dataset,
    location,
    context_width,
    context_height,
    output_height_pixels,
    output_width_pixels,
):
    xr_context: xr.Dataset = _get_spatial_crop(
        xr_data,
        location=location,
        roi_width_meters=context_width,
        roi_height_meters=context_height,
    )

    # Resamples to the same number of pixels for both center and contexts
    xr_context = _resample_to_pixel_size(xr_context, output_height_pixels, output_width_pixels)
    return xr_context


def _get_spatial_crop(xr_data, location, roi_height_meters: int, roi_width_meters: int):
    xr_coords, xr_x_dim, xr_y_dim = spatial_coord_type(xr_data)

    # Compute the index for left and right:
    half_height = roi_height_meters // 2
    half_width = roi_width_meters // 2

    # Find the bounding box values for the location in either lat-lon or OSGB coord systems
    if location.coordinate_system == "lat_lon":
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

    (left, right), (bottom, top) = convert_coords_to_match_xarray(
        x=np.array([left, right], dtype=np.float32),
        y=np.array([bottom, top], dtype=np.float32),
        from_coords=location.coordinate_system,
        xr_data=xr_data,
    )

    # Select a patch from the xarray data
    x_mask = (left <= xr_data[xr_x_dim]) & (xr_data[xr_x_dim] <= right)
    y_mask = (bottom <= xr_data[xr_y_dim]) & (xr_data[xr_y_dim] <= top)
    selected = xr_data.isel({xr_x_dim: x_mask, xr_y_dim: y_mask})

    return selected


def _resample_to_pixel_size(xr_data, height_pixels, width_pixels) -> np.ndarray:
    if "x_geostationary" in xr_data.dims:
        x_coords = xr_data["x_geostationary"].values
        y_coords = xr_data["y_geostationary"].values
    elif "x_osgb" in xr_data.dims:
        x_coords = xr_data["x_osgb"].values
        y_coords = xr_data["y_osgb"].values
    else:
        x_coords = xr_data["x"].values
        y_coords = xr_data["y"].values
    # Resample down to the number of pixels wanted
    x_coords = np.linspace(x_coords[0], x_coords[-1], num=width_pixels)
    y_coords = np.linspace(y_coords[0], y_coords[-1], num=height_pixels)
    if "x_geostationary" in xr_data.dims:
        xr_data = xr_data.interp(
            x_geostationary=x_coords, y_geostationary=y_coords, method="linear"
        )
    elif "x_osgb" in xr_data.dims:
        xr_data = xr_data.interp(x_osgb=x_coords, y_osgb=y_coords, method="linear")
    else:
        xr_data = xr_data.interp(x=x_coords, y=y_coords, method="linear")
    # Extract just the data now
    return xr_data


def _create_time_image(xr_data, time_dim: str, output_height_pixels: int, output_width_pixels: int):
    # Create trig decomposition of datetime values, tiled over output height and width
    datetimes = xr_data[time_dim].values
    trig_decomposition = trigonometric_datetime_transformation(datetimes)
    tiled_data = np.expand_dims(trig_decomposition, (2, 3))
    tiled_data = np.tile(tiled_data, (1, 1, output_height_pixels, output_width_pixels))
    return tiled_data


def _create_sun_image(image_xr, x_dim, y_dim, time_dim, normalize):
    # Create empty image to use for the PV Systems, assumes image has x and y coordinates
    sun_image = np.zeros(
        (
            2,  # Azimuth and elevation
            len(image_xr[y_dim]),
            len(image_xr[x_dim]),
            len(image_xr[time_dim]),
        ),
        dtype=np.float32,
    )
    if "geostationary" in x_dim:
        lons, lats = geostationary_area_coords_to_lonlat(
            x=image_xr[x_dim].values, y=image_xr[y_dim].values, xr_data=image_xr
        )
    else:
        lons, lats = osgb_to_lon_lat(x=image_xr.x_osgb.values, y=image_xr.y_osgb.values)
    time_utc = image_xr[time_dim].values

    # Loop round each example to get the Sun's elevation and azimuth:
    # Go through each time on its own, lat lons still in order of image
    # TODO Make this faster
    # dt = pd.DatetimeIndex(dt)  # pvlib expects a `pd.DatetimeIndex`.
    for example_idx, (lat, lon) in enumerate(zip(lats, lons)):
        solpos = pvlib.solarposition.get_solarposition(
            time=time_utc,
            latitude=lat,
            longitude=lon,
            # Which `method` to use?
            # pyephem seemed to be a good mix between speed and ease but causes segfaults!
            # nrel_numba doesn't work when using multiple worker processes.
            # nrel_c is probably fastest but requires C code to be manually compiled:
            # https://midcdmz.nrel.gov/spa/
        )
        sun_image[0][:][example_idx] = solpos["azimuth"]
        sun_image[1][example_idx][:] = solpos["elevation"]

    # Flip back to normal ordering
    sun_image = np.transpose(sun_image, [3, 0, 1, 2])

    # Normalize.
    if normalize:
        sun_image[:, 0] = (sun_image[:, 0] - AZIMUTH_MEAN) / AZIMUTH_STD
        sun_image[:, 1] = (sun_image[:, 1] - ELEVATION_MEAN) / ELEVATION_STD
    return sun_image
