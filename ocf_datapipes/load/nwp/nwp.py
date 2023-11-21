"""NWP Loader"""
import logging
from pathlib import Path
from typing import Union

import xarray as xr
from ocf_blosc2 import Blosc2  # noqa: F401
from torch.utils.data import IterDataPipe, functional_datapipe

from ocf_datapipes.load.nwp.providers.ecmwf import open_ifs
from ocf_datapipes.load.nwp.providers.gfs import open_gfs
from ocf_datapipes.load.nwp.providers.icon import open_icon_eu, open_icon_global
from ocf_datapipes.load.nwp.providers.ukv import open_ukv

_log = logging.getLogger(__name__)


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
            convert_to_lat_lon: Whether to convert to lat/lon, or leave in native format
                i.e. OSGB for UKV, Lat/Lon for ICON EU, Icoshedral grid for ICON Global
        """
        self.zarr_path = zarr_path
        if provider.lower() == "ukv" or provider == "UKMetOffice":
            self.open_nwp = open_ukv
        elif provider.lower() == "icon-eu":
            self.open_nwp = open_icon_eu
        elif provider.lower() == "icon-global":
            self.open_nwp = open_icon_global
        elif provider.lower() == "ecmwf":
            self.open_nwp = open_ifs
        elif provider.lower() == "gfs":
            self.open_nwp = open_gfs
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        """Opens the NWP data"""
        _log.debug("Opening NWP data: %s", self.zarr_path)
        nwp = self.open_nwp(self.zarr_path)
        while True:
            yield nwp


@functional_datapipe("open_latest_nwp")
class OpenLatestNWPDataPipe(IterDataPipe):
    """Yields the most recent observation from NWP data"""

    def __init__(self, base_nwp_datapipe: OpenNWPIterDataPipe) -> None:
        """Selects most recent observation from NWP data

        Args:
            base_nwp_datapipe (OpenNWPIterDataPipe): Base DataPipe, opening zarr
        """
        self.base_nwp_datapipe = base_nwp_datapipe

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        """Selects most recent entry

        Returns:
            Union[xr.DataArray, xr.Dataset]: NWP slice

        Yields:
            Iterator[Union[xr.DataArray, xr.Dataset]]: Iterator of most recent NWP data
        """
        for nwp_data in self.base_nwp_datapipe:
            _nwp = nwp_data.sel(init_time_utc=nwp_data.init_time_utc.max())
            time = _nwp.init_time_utc.values
            _log.debug(f"Selected most recent NWP observation, at: {time}")
            yield _nwp
