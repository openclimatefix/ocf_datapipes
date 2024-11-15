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
    ):
        """
        Opens NWP Zarr and yields it

        Args:
            zarr_path: Path to the Zarr file
            provider: NWP provider
        """
        self.zarr_path = zarr_path
        logger.info(f"Using {provider.lower()}")
        if provider.lower() == "ukv":
            self.open_nwp = open_ukv
        elif provider.lower() == "icon-eu":
            self.open_nwp = open_icon_eu
        elif provider.lower() == "icon-global":
            self.open_nwp = open_icon_global
        elif provider.lower() in ("ecmwf", "mo_global"):  # same schema so using the same loader
            self.open_nwp = open_ifs
        elif provider.lower() == "gfs":
            self.open_nwp = open_gfs
        elif provider.lower() == "excarta":
            self.open_nwp = open_excarta
        elif provider.lower() == "merra2":
            self.open_nwp = open_merra2
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        """Opens the NWP data"""
        logger.debug("Opening NWP data: %s", self.zarr_path)
        nwp = self.open_nwp(self.zarr_path)
        while True:
            yield nwp
