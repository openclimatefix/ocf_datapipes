"""NWP Loader"""

import logging
from pathlib import Path
from typing import Union

import xarray as xr
from ocf_blosc2 import Blosc2  # noqa: F401
from torch.utils.data import IterDataPipe, functional_datapipe

from ocf_datapipes.load.nwp.providers.ecmwf import open_ifs
from ocf_datapipes.load.nwp.providers.excarta import open_excarta
from ocf_datapipes.load.nwp.providers.gfs import open_gfs
from ocf_datapipes.load.nwp.providers.icon import open_icon_eu, open_icon_global
from ocf_datapipes.load.nwp.providers.merra2 import open_merra2
from ocf_datapipes.load.nwp.providers.ukv import open_ukv

logger = logging.getLogger(__name__)


@functional_datapipe("open_nwp")
class OpenNWPIterDataPipe(IterDataPipe):
    """Opens NWP Zarr and yields it"""

    def __init__(
        self,
        zarr_path: Union[Path, str, list[Path], list[str]],
        provider: str = "ukv",
        check_for_zeros: bool = False,
        check_physical_limits: bool = False,
    ):
        """
        Opens NWP Zarr and yields it

        Args:
            zarr_path: Path to the Zarr file
            provider: NWP provider
            check_for_zeros: Check for zeros in the NWP data
            check_physical_limits: Check the physical limits of nwp data (e.g. -100<temperature<100)
        """
        self.zarr_path = zarr_path
        self.check_for_zeros = check_for_zeros
        self.check_physical_limits = check_physical_limits
        self.limits = {
            "temperature": (-100, 60),  # Celsius
            "specific_humidity": (0, 0.03),  # kg/kg
            "relative_humidity": (0, 100),  # Percentage
            "pressure": (0, 1100),  # hPa (sea level pressure)
            "u_wind": (-200, 200),  # m/s
            "v_wind": (-200, 200),  # m/s
            "geopotential": (0, 100000),  # m^2/s^2
            "total_precipitation": (0, 2000),  # mm/day
            "convective_precipitation": (0, 1000),  # mm/day
            "snowfall": (0, 1000),  # mm water equivalent/day
            "graupel": (0, 500),  # mm water equivalent/day
            "cloud_cover": (0, 100),  # Percentage
            "surface_temperature": (-90, 60),  # Celsius
            "sea_surface_temperature": (-2, 35),  # Celsius
            "soil_temperature": (-50, 60),  # Celsius
            "soil_moisture": (0, 1),  # m^3/m^3
            "visibility": (0, 100000),  # meters
            "wind_gust": (0, 250),  # m/s
            "solar_radiation": (0, 1500),  # W/m^2
            "longwave_radiation": (0, 750),  # W/m^2
            "evaporation": (0, 50),  # mm/day
            "potential_evaporation": (0, 100),  # mm/day
            "boundary_layer_height": (0, 5000),  # meters
            "cape": (0, 10000),  # J/kg
            "cin": (0, 1000),  # J/kg
            "lifted_index": (-15, 15),  # Kelvin
            "total_column_water": (0, 100),  # kg/m^2
            "ozone_concentration": (0, 1000),  # Dobson units
            "dew_point_temperature": (-100, 35),  # Celsius
            "wet_bulb_temperature": (-100, 35),  # Celsius
            "potential_temperature": (0, 1000),  # Kelvin
            "equivalent_potential_temperature": (0, 1000),  # Kelvin
            "vorticity": (-1e-3, 1e-3),  # 1/s
            "divergence": (-1e-3, 1e-3),  # 1/s
            "vertical_velocity": (-50, 50),  # m/s
            "cloud_base_height": (0, 20000),  # meters
            "cloud_top_height": (0, 20000),  # meters
            "cloud_water_content": (0, 5),  # g/kg
            "ice_water_content": (0, 5),  # g/kg
            "surface_roughness": (0, 10),  # meters
            "albedo": (0, 1),  # dimensionless
            "friction_velocity": (0, 5),  # m/s
            "sensible_heat_flux": (-500, 500),  # W/m^2
            "latent_heat_flux": (-500, 500),  # W/m^2
            "momentum_flux": (-10, 10),  # N/m^2
            "surface_pressure": (300, 1100),  # hPa
            "mean_sea_level_pressure": (870, 1090),  # hPa
            "tropopause_pressure": (50, 500),  # hPa
            "tropopause_temperature": (-100, 0),  # Celsius
            "precipitable_water": (0, 100),  # mm
            "total_cloud_cover": (0, 100),  # Percentage
            "low_cloud_cover": (0, 100),  # Percentage
            "medium_cloud_cover": (0, 100),  # Percentage
            "high_cloud_cover": (0, 100),  # Percentage
            "convective_available_potential_energy": (0, 10000),  # J/kg
            "convective_inhibition": (0, 1000),  # J/kg
            "storm_relative_helicity": (-1000, 1000),  # m^2/s^2
            "bulk_richardson_number": (-10, 10),  # dimensionless
            "lifted_condensation_level": (0, 5000),  # meters
            "level_of_free_convection": (0, 20000),  # meters
            "equilibrium_level": (0, 20000),  # meters
            "UKV": (250, 330),  # UKV specific
        }
        logger.info(f"Using {provider.lower()}")
        if provider.lower() == "ukv":
            self.open_nwp = open_ukv
        elif provider.lower() == "icon-eu":
            self.open_nwp = open_icon_eu
        elif provider.lower() == "icon-global":
            self.open_nwp = open_icon_global
        elif provider.lower() == "ecmwf":
            self.open_nwp = open_ifs
        elif provider.lower() == "gfs":
            self.open_nwp = open_gfs
        elif provider.lower() == "excarta":
            self.open_nwp = open_excarta
        elif provider.lower() == "merra2":
            self.open_nwp = open_merra2
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:  # type: ignore
        """Opens the NWP data"""
        logger.debug("Opening NWP data: %s", self.zarr_path)
        nwp = self.open_nwp(self.zarr_path)
        if self.check_for_zeros:
            self.check_if_zeros(nwp)
        if self.check_physical_limits:
            self.check_if_physical_limits(nwp)
        while True:
            yield nwp

    def check_if_zeros(self, nwp: Union[xr.DataArray, xr.Dataset]):
        """Checks if the NWP data contains zeros"""
        if isinstance(nwp, xr.DataArray):
            if (nwp.values == 0).any():
                raise ValueError(
                    f"NWP DataArray contains{(nwp.values == 0).sum()*100/nwp.values.size}% zeros"
                )
        if isinstance(nwp, xr.Dataset):
            for var in nwp:
                if (nwp[var].values == 0).any():
                    raise ValueError(
                        f"NWP Dataset variable{var} "
                        f"contains {(nwp[var].values == 0).sum()*100/nwp[var].values.size}% zeros"
                    )

    def check_if_physical_limits(self, nwp: Union[xr.DataArray, xr.Dataset]):
        """Checks if the NWP data is within physical limits"""
        if isinstance(nwp, xr.DataArray):
            var_name = nwp.name
            if var_name in self.limits:
                lower, upper = self.limits[var_name]
                if (nwp < lower).any() or (nwp > upper).any():
                    raise ValueError(
                        f"NWP data {var_name} is outside physical limits: ({lower},{upper})"
                    )
        elif isinstance(nwp, xr.Dataset):
            for var_name, (lower, upper) in self.limits.items():
                if var_name in nwp.variables:
                    if not ((nwp[var_name] >= lower).all() and (nwp[var_name] <= upper).all()):
                        raise ValueError(
                            f"NWP data {var_name} is outside physical limits: ({lower},{upper})"
                        )
