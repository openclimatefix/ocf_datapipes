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
from ocf_datapipes.utils.geospatial import load_geostationary_area_definition_and_transform_osgb

logger = logging.getLogger(__name__)


@functional_datapipe("create_pv_image")
class CreatePVImageIterDataPipe(IterDataPipe):
    """Create PV image from individual sites"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        image_datapipe: IterDataPipe,
        normalize: bool = False,
        image_dim: str = "geostationary",
        max_num_pv_systems: int = -1,
        always_return_first: bool = False,
        seed: int = None,
        take_n_pv_values_per_pixel: int = -1,
        normalize_by_pvlib: bool = False,
    ):
        """
        Creates a 3D data cube of PV output image x number of timesteps

        TODO Include PV System IDs or something, currently just takes the outputs

        Args:
            source_datapipe: Source datapipe of PV data
            image_datapipe: Datapipe emitting images to get the shape from, with coordinates
            normalize: Whether to normalize based off the image max, or leave raw data
            image_dim: Dimension name for the x and y dimensions
            max_num_pv_systems: Max number of PV systems to use
            always_return_first: Always return the first image data cube, to save computation
                Only use for if making the image at the beginning of the stack
            seed: Random seed to use if using max_num_pv_systems
            take_n_pv_values_per_pixel: Take N PV values as the value for the pixel
                If -1, averages the PV generation if there are multiple PVs per pixel.
            normalize_by_pvlib: Normalize by pvlib's poa_global based off
                tilt/orientation/capacity/lat/lon of the system

        """
        if normalize_by_pvlib is not False or normalize is not False:
            assert normalize != normalize_by_pvlib, ValueError(
                "Cannot normalize by both max, and pvlib"
            )
        self.source_datapipe = source_datapipe
        self.image_datapipe = image_datapipe
        self.normalize = normalize
        self.x_dim = "x_" + image_dim
        self.y_dim = "y_" + image_dim
        self.max_num_pv_systems = max_num_pv_systems
        self.rng = np.random.default_rng(seed=seed)
        self.always_return_first = always_return_first
        self.take_n_pv_values_per_pixel = take_n_pv_values_per_pixel
        self.normalize_by_pvlib = normalize_by_pvlib

    def __iter__(self) -> xr.DataArray:
        for pv_systems_xr, image_xr in Zipper(self.source_datapipe, self.image_datapipe):
            if 0 < self.max_num_pv_systems <= len(pv_systems_xr.pv_system_id.values):
                subset_of_pv_system_ids = self.rng.choice(
                    pv_systems_xr.pv_system_id,
                    size=self.max_num_pv_systems,
                    replace=False,
                )
                pv_systems_xr = pv_systems_xr.sel(pv_system_id=subset_of_pv_system_ids)

            if "geostationary" in self.x_dim:
                _osgb_to_geostationary = load_geostationary_area_definition_and_transform_osgb(
                    image_xr
                )
            # Create empty image to use for the PV Systems, assumes image has x and y coordinates
            pv_image = np.zeros(
                (
                    len(pv_systems_xr["time_utc"]),
                    len(image_xr[self.y_dim]),
                    len(image_xr[self.x_dim]),
                ),
                dtype=np.float32,
            )
            pv_position_dict = defaultdict(list)
            for i, pv_system_id in enumerate(pv_systems_xr["pv_system_id"]):
                try:
                    # went for isel incase there is a duplicated pv_system_id
                    pv_system = pv_systems_xr.isel(pv_system_id=i)
                except Exception as e:
                    logger.warning(
                        f"Could not select {pv_system_id} " f"from {pv_systems_xr.pv_system_id}"
                    )
                    raise e
                if "geostationary" in self.x_dim:
                    pv_x, pv_y = _osgb_to_geostationary(
                        xx=pv_system["x_osgb"].values, yy=pv_system["y_osgb"].values
                    )
                else:
                    pv_x = pv_system["x_osgb"]
                    pv_y = pv_system["y_osgb"]
                # Quick check as search sorted doesn't give an error if it is not in the range
                if pv_x < image_xr[self.x_dim][0].values or pv_x > image_xr[self.x_dim][-1].values:
                    continue
                # Y Coordinates are in reverse for satellite data
                if pv_y > image_xr[self.y_dim][0].values or pv_y < image_xr[self.y_dim][-1].values:
                    continue
                if "geostationary" in self.x_dim:
                    x_idx = np.searchsorted(image_xr[self.x_dim].values, pv_x) - 1
                    # y_geostationary is in descending order:
                    y_idx = len(image_xr[self.y_dim]) - (
                        np.searchsorted(image_xr[self.y_dim].values[::-1], pv_y) - 1
                    )
                else:
                    x_idx = np.searchsorted(pv_x, image_xr[self.x_dim])
                    y_idx = np.searchsorted(pv_y, image_xr[self.y_dim])

                # Add location to same one, so know if multiple overlap
                # Filter if normalizing by pvlib,
                # only include ones that have tilt and orientation as numbers
                if self.normalize_by_pvlib:
                    if not np.isfinite(pv_system.orientation.values) and not np.isfinite(
                        pv_system.tilt.values
                    ):
                        continue
                pv_position_dict[(y_idx, x_idx)].append(pv_system)
            for location, system_list in pv_position_dict.items():
                y_idx, x_idx = location[0], location[1]
                if 0 < self.take_n_pv_values_per_pixel < len(system_list):
                    system_numbers = self.rng.choice(
                        list(range(len(system_list))),
                        size=self.take_n_pv_values_per_pixel,
                        replace=False,
                    )
                    system_list = [system_list[idx] for idx in system_numbers]
                avg_generation = np.zeros_like(system_list[0].values)
                for pv_system in system_list:
                    if self.normalize_by_pvlib:
                        pv_system = _normalize_by_pvlib(pv_system)
                    avg_generation += np.nan_to_num(pv_system.values)
                avg_generation /= len(system_list)
                pv_image[:, y_idx, x_idx] = avg_generation

            if self.normalize:
                if np.nanmax(pv_image) > 0:
                    pv_image /= np.nanmax(pv_image)
            pv_image = np.nan_to_num(pv_image)

            # Should return Xarray as in Xarray transforms
            # Same coordinates as the image xarray, so can take that
            pv_image = _create_data_array_from_image(pv_image, pv_systems_xr, image_xr)
            if self.always_return_first:
                while True:
                    yield pv_image
            yield pv_image


def _create_data_array_from_image(
    pv_image: np.ndarray,
    pv_systems_xr: Union[xr.Dataset, xr.DataArray],
    image_xr: Union[xr.Dataset, xr.DataArray],
):
    data_array = xr.DataArray(
        data=pv_image,
        coords=(
            ("time_utc", pv_systems_xr.time_utc.values),
            ("y_geostationary", image_xr.y_geostationary.values),
            ("x_geostationary", image_xr.x_geostationary.values),
        ),
        name="pv_image",
    ).astype(np.float32)
    data_array.attrs = image_xr.attrs
    return data_array


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


def _normalize_by_pvlib(pv_system):
    """
    Normalize the output by pv_libs poa_global

    Args:
        pv_system: PV System in Xarray DataArray

    Returns:
        PV System in xarray DataArray, but normalized values
    """
    # TODO Add elevation
    pvlib_loc = pvlib.location.Location(
        latitude=pv_system.latitude.values, longitude=pv_system.longitude.values
    )
    times = pd.DatetimeIndex(pv_system.time_utc.values)
    solar_position = pvlib_loc.get_solarposition(times=times)
    clear_sky = pvlib_loc.get_clearsky(times)
    total_irradiance = pvlib.irradiance.get_total_irradiance(
        pv_system.tilt.values,
        pv_system.orientation.values,
        solar_zenith=solar_position["zenith"],
        solar_azimuth=solar_position["azimuth"],
        dni=clear_sky["dni"],
        dhi=clear_sky["dhi"],
        ghi=clear_sky["ghi"],
    )
    # Guess want fraction of total irradiance on panel, to get fraction to do with capacity
    fraction_clear_sky = total_irradiance["poa_global"] / (
        clear_sky["dni"] + clear_sky["dhi"] + clear_sky["ghi"]
    )
    pv_system /= pv_system.capacity_watt_power
    pv_system *= fraction_clear_sky
    return pv_system
