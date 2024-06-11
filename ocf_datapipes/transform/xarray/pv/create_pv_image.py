"""Convert point PV sites to image output"""

import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import pvlib
import xarray as xr
from torch.utils.data import IterDataPipe, functional_datapipe

from ocf_datapipes.utils import Zipper
from ocf_datapipes.utils.geospatial import (
    lon_lat_to_geostationary_area_coords,
    osgb_to_geostationary_area_coords,
    spatial_coord_type,
)

logger = logging.getLogger(__name__)


@functional_datapipe("create_pv_image")
class CreatePVImageIterDataPipe(IterDataPipe):
    """Create PV image from individual sites"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        image_datapipe: IterDataPipe,
        normalize: bool = False,
        max_num_pv_systems: int = np.inf,
        always_return_first: bool = False,
        seed: int = None,
        max_num_pv_systems_per_pixel: int = np.inf,
        normalize_by_pvlib: bool = False,
    ):
        """
        Creates a 3D data cube of PV output image x number of timesteps

        TODO Include PV System IDs or something, currently just takes the outputs

        Args:
            source_datapipe: Source datapipe of PV data
            image_datapipe: Datapipe emitting images to get the shape from, with coordinates
            normalize: Whether to normalize based off the image max, or leave raw data
            max_num_pv_systems: Max number of PV systems to consider to generate entire image
            always_return_first: Always return the first image data cube, to save computation
                Only use for if making the image at the beginning of the stack
            seed: Random seed to use if using max_num_pv_systems
            max_num_pv_systems_per_pixel: Limit the number of PV systems used when finding the
                generation timeseries in each pixel
            normalize_by_pvlib: Normalize by pvlib's poa_global based off
                tilt/orientation/capacity/lat/lon of the system

        """
        if normalize_by_pvlib and normalize:
            raise ValueError("Cannot normalize by both max, and pvlib")
        assert max_num_pv_systems_per_pixel > 0
        assert max_num_pv_systems > 0

        self.source_datapipe = source_datapipe
        self.image_datapipe = image_datapipe
        self.normalize = normalize
        self.max_num_pv_systems = max_num_pv_systems
        self.rng = np.random.default_rng(seed=seed)
        self.always_return_first = always_return_first
        self.max_num_pv_systems_per_pixel = max_num_pv_systems_per_pixel
        self.normalize_by_pvlib = normalize_by_pvlib

    def __iter__(self) -> xr.DataArray:
        for pv_systems_xr, image_xr in Zipper(self.source_datapipe, self.image_datapipe):
            pv_coords, pv_x_dim, pv_y_dim = spatial_coord_type(pv_systems_xr)
            image_coords, image_x_dim, image_y_dim = spatial_coord_type(image_xr)

            # Randomly sample systems if too many
            if self.max_num_pv_systems <= len(pv_systems_xr.pv_system_id):
                subset_of_pv_system_ids = self.rng.choice(
                    pv_systems_xr.pv_system_id,
                    size=self.max_num_pv_systems,
                    replace=False,
                )
                pv_systems_xr = pv_systems_xr.sel(pv_system_id=subset_of_pv_system_ids)

            # Find and store spatial index positions of the PV systems within the target image
            pv_position_dict = defaultdict(list)

            for i, pv_system_id in enumerate(pv_systems_xr["pv_system_id"]):
                # went for isel incase there is a duplicated pv_system_id
                pv_system = pv_systems_xr.isel(pv_system_id=i)

                if pv_coords == "osgb" and image_coords == "geostationary":
                    pv_x, pv_y = osgb_to_geostationary_area_coords(
                        x=pv_system["x_osgb"].values,
                        y=pv_system["y_osgb"].values,
                        xr_data=image_xr,
                    )
                elif pv_coords == "lon_lat" and image_coords == "geostationary":
                    pv_x, pv_y = lon_lat_to_geostationary_area_coords(
                        x=pv_system["longitude"].values,
                        y=pv_system["latitude"].values,
                        xr_data=image_xr,
                    )

                else:
                    raise NotImplementedError(
                        f"PV coords of type {pv_coords} and image coords of type {image_coords}"
                    )

                # Check the PV system is within the image
                in_range = (image_xr[image_x_dim].min() < pv_x < image_xr[image_x_dim].max()) and (
                    image_xr[image_y_dim].min() < pv_y < image_xr[image_y_dim].max()
                )

                # If normalizing by pvlib we need tilt and orientation values
                is_normalizable = (not self.normalize_by_pvlib) or np.isfinite(
                    [pv_system.orientation.values, pv_system.tilt.values]
                ).all()

                if not in_range or not is_normalizable:
                    # Skip the PV system if not inside the image
                    # Skip the PV system if pvlib normalizarion requested but cannot be completed
                    continue

                x_idx = image_xr.get_index(image_x_dim).get_indexer([pv_x], method="nearest")[0]
                y_idx = image_xr.get_index(image_y_dim).get_indexer([pv_y], method="nearest")[0]

                pv_position_dict[(y_idx, x_idx)].append(pv_system)

            # Create empty image to use for the PV Systems, assumes image has x and y coordinates
            pv_image = np.zeros(
                (
                    len(pv_systems_xr["time_utc"]),
                    len(image_xr[image_y_dim]),
                    len(image_xr[image_x_dim]),
                ),
                dtype=np.float32,
            )

            for (y_idx, x_idx), system_list in pv_position_dict.items():
                # Randomly sample systems if too many per pixel
                if self.max_num_pv_systems_per_pixel < len(system_list):
                    random_inds = self.rng.choice(
                        np.arange(len(system_list)),
                        size=self.max_num_pv_systems_per_pixel,
                        replace=False,
                    )
                    system_list = [system_list[i] for i in random_inds]

                # Find average output timeseries of all systems in pixel
                avg_generation = np.zeros_like(system_list[0].values)
                for pv_system in system_list:
                    if self.normalize_by_pvlib:
                        pv_system = _normalize_by_pvlib(pv_system)
                    avg_generation += np.nan_to_num(pv_system.values)
                avg_generation /= len(system_list)
                pv_image[:, y_idx, x_idx] = avg_generation

            if self.normalize:
                max_val = np.nanmax(pv_image)
                if max_val > 0:
                    pv_image = pv_image / max_val

            # Should return xarray
            # Same coordinates as the image xarray, so can take that
            pv_image_xr = xr.DataArray(
                data=pv_image,
                coords=(
                    ("time_utc", pv_systems_xr.time_utc.values),
                    (image_y_dim, image_xr[image_y_dim].values),
                    (image_x_dim, image_xr[image_x_dim].values),
                ),
                name="pv_image",
            )
            pv_image_xr.attrs = image_xr.attrs

            if self.always_return_first:
                while True:
                    yield pv_image_xr
            yield pv_image_xr


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
    pv_system /= pv_system.observed_capacity_wp
    pv_system *= fraction_clear_sky
    return pv_system
