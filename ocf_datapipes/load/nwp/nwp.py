"""NWP Loader"""

import logging
from pathlib import Path
from typing import Union

import dask
import dask.array
import numpy as np
import xarray as xr
from ocf_blosc2 import Blosc2  # noqa: F401
from torch.utils.data import IterDataPipe, functional_datapipe

from ocf_datapipes.load.nwp.providers.ecmwf import open_ifs
from ocf_datapipes.load.nwp.providers.excarta import open_excarta
from ocf_datapipes.load.nwp.providers.gfs import open_gfs
from ocf_datapipes.load.nwp.providers.icon import open_icon_eu, open_icon_global
from ocf_datapipes.load.nwp.providers.merra2 import open_merra2
from ocf_datapipes.load.nwp.providers.ukv import open_ukv

from .constants import NWP_LIMITS

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
        check_for_nans: bool = False,
    ):
        """
        Opens NWP Zarr and yields it

        Args:
            zarr_path: Path to the Zarr file
            provider: NWP provider
            check_for_zeros: Check for zeros in the NWP data
            check_physical_limits: Check the physical limits of nwp data (e.g. -100<temperature<100)
            check_for_nans: Check for NaNs in the NWP data
        """
        self.zarr_path = zarr_path
        self.check_for_zeros = check_for_zeros
        self.check_physical_limits = check_physical_limits
        self.check_for_nans = check_for_nans
        self.limits = NWP_LIMITS

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
        if self.check_for_nans:
            self.check_if_nans(nwp)
        while True:
            yield nwp

    def check_if_zeros(self, nwp: Union[xr.DataArray, xr.Dataset]):
        """Checks if the NWP data contains zeros"""

        def count_zeros(block):
            return (block == 0).sum()

        def check_zeros(result):
            if result > 0:
                raise ValueError(f"NWP data contains {result*100/nwp.size}% zeros")

        if isinstance(nwp, xr.DataArray):
            if dask.is_dask_collection(nwp.data):
                zero_count = nwp.data.map_blocks(count_zeros, dtype=int).compute()
                check_zeros(zero_count)
            else:
                if (nwp.values == 0).any():
                    raise ValueError(
                        f"NWP DataArray contains{(nwp.values == 0).sum()*100/nwp.values.size}% "
                        "zeros"
                    )
        elif isinstance(nwp, xr.Dataset):
            for var in nwp:
                if dask.is_dask_collection(nwp[var].data):
                    zero_count = nwp[var].data.map_blocks(count_zeros, dtype=int).compute()
                    check_zeros(zero_count)
                else:
                    if (nwp[var].values == 0).any():
                        raise ValueError(
                            f"NWP Dataset variable{var} "
                            f"contains {(nwp[var].values == 0).sum()*100/nwp[var].values.size}% "
                            "zeros"
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
    def check_if_nans(self, nwp: Union[xr.DataArray, xr.Dataset]):
        """Checks if the NWP data contains NaNs"""
        if isinstance(nwp, xr.DataArray):
            if dask.is_dask_collection(nwp.data):
                if dask.array.isnan(nwp.data).any().compute():
                    raise ValueError("NWP data contains NaNs")
            else:
                if np.isnan(nwp.data).any():
                    raise ValueError("NWP DataArray contains NaNs")
        elif isinstance(nwp, xr.Dataset):
            for var in nwp.data_vars:
                if dask.is_dask_collection(nwp[var].data):
                    if dask.array.isnan(nwp[var].data).any().compute():
                        raise ValueError(f"NWP Dataset variable{var} contains NaNs")
                else:
                    if np.isnan(nwp[var].data).any():
                        raise ValueError(f"NWP Dataset variable{var} contains NaNs")
