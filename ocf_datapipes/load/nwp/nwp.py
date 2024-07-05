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

        # limits for NWP data in accordance with https://huggingface.co/openclimatefix/pvnet_uk_region/blob/main/data_config.yaml
        self.limits = {
            "t2m": (200, 350),  # Temperature in Kelvin (-100°C to 60°C)
            "dswrf": (0, 1500),  # Downward short-wave radiation flux, W/m^2
            "dlwrf": (0, 750),  # Downward long-wave radiation flux, W/m^2
            "hcc": (0, 100),  # High cloud cover, %
            "mcc": (0, 100),  # Medium cloud cover, %
            "lcc": (0, 100),  # Low cloud cover, %
            "tcc": (0, 100),  # Total cloud cover, %
            "sde": (0, 1000),  # Snowfall depth, meters
            "sr": (0, 10),  # Surface roughness, meters
            "duvrs": (0, 500),  # Direct UV radiation at surface, W/m^2 (positive values only)
            "u10": (-200, 200),  # U component of 10m wind, m/s
            "v10": (-200, 200),  # V component of 10m wind, m/s
            # UKV NWP channels (additional to ECMWF)
            "prate": (0, 2000),  # Precipitation rate, , kg/m^2/s (equivalent to 0-2000 mm/day)
            "r": (0, 100),  # Relative humidity, %
            "si10": (0, 250),  # Wind speed at 10m, m/s
            "t": (200, 350),  # Temperature in Kelvin (-100°C to 60°C)
            "vis": (0, 100000),  # Visibility, meters
            # Satellite channels (no direct mapping to physical limits, using placeholder values)
            "IR_016": (0, 1000),  # Infrared channel
            "IR_039": (0, 1000),  # Infrared channel
            "IR_087": (0, 1000),  # Infrared channel
            "IR_097": (0, 1000),  # Infrared channel
            "IR_108": (0, 1000),  # Infrared channel
            "IR_120": (0, 1000),  # Infrared channel
            "IR_134": (0, 1000),  # Infrared channel
            "VIS006": (0, 1000),  # Visible channel
            "VIS008": (0, 1000),  # Visible channel
            "WV_062": (0, 1000),  # Water vapor channel
            "WV_073": (0, 1000),  # Water vapor channel
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
            var_name = nwp.channel.values[0]
            if var_name in self.limits:
                lower, upper = self.limits[var_name]
                if (nwp < lower).any() or (nwp > upper).any():
                    raise ValueError(
                        f"NWP data {var_name} is outside physical limits: ({lower},{upper})"
                    )
        elif isinstance(nwp, xr.Dataset):
            for var_name, (lower, upper) in self.limits.items():
                if var_name in nwp.channel:
                    if not ((nwp[var_name] >= lower).all() and (nwp[var_name] <= upper).all()):
                        raise ValueError(
                            f"NWP data {var_name} is outside physical limits: ({lower},{upper})"
                        )
