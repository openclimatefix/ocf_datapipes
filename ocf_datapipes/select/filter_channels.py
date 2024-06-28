"""Select channels"""

import logging
from typing import List, Union

import numpy as np
import xarray as xr
from torch.utils.data import IterDataPipe, functional_datapipe

logger = logging.getLogger(__name__)


@functional_datapipe("filter_channels")
class FilterChannelsIterDataPipe(IterDataPipe):
    """Filter channels"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        channels: List[str],
        dim_name: str = "channel",
        provider: str = None,
    ):
        """
        Filter channels

        Args:
            source_datapipe: Datapipe of Xarray objects
            channels: Channel names to keep
            dim_name: Dimension name for channels
            provider: Name of NWP source, if available
        """
        self.source_datapipe = source_datapipe
        self.channels = channels
        self.dim_name = dim_name
        self.provider = provider

        if self.provider == "gfs":
            flux_vars = np.intersect1d(self.channels, ["dswrf", "dlwrf"])

            if len(flux_vars) > 0:
                logger.warning(
                    f"You have requested channels that have no step 0: {flux_vars}. "
                    f"Step 0 will be set to NaN. "
                    f"For more info see https://github.com/openclimatefix/ocf_datapipes/issues/253"
                )

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        for xr_data in self.source_datapipe:
            if "channel" not in xr_data.dims and isinstance(
                xr_data, xr.Dataset
            ):  # Variables are in their own data variables, not channels
                logger.debug(
                    f"Selecting Channels: {self.channels} out of {xr_data.data_vars.keys()}"
                )
                # Select data variables from dataset that are in channels
                xr_data = xr_data[self.channels]
                xr_data.coords[self.dim_name] = self.channels
            else:
                logger.debug(
                    f"Selecting Channels: {self.channels} out of {xr_data['channel'].values}"
                )
                xr_data = xr_data.sel({self.dim_name: list(self.channels)})
            yield xr_data
