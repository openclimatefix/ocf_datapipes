"""Select channels"""
import logging
from typing import List, Union

import xarray as xr
from torch.utils.data import IterDataPipe, functional_datapipe

logger = logging.getLogger(__name__)


@functional_datapipe("select_channels")
class SelectChannelsIterDataPipe(IterDataPipe):
    """Select channels"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        channels: List[str],
        dim_name: str = "channel",
    ):
        """
        Select channels

        Args:
            source_datapipe: Datapipe of Xarray objects
            channels: Channel names to keep
            dim_name: Dimension name for time
        """
        self.source_datapipe = source_datapipe
        self.channels = channels
        self.dim_name = dim_name

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        for xr_data in self.source_datapipe:
            if "channels" not in xr_data.dims and isinstance(
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
