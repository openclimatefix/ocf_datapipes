"""Select channels"""
import logging
from typing import List, Union

import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

logger = logging.getLogger(__name__)


@functional_datapipe("select_channels")
class SelectChannelsIterDataPipe(IterDataPipe):
    """Select channels"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        channels: List[str],
        dim_name: str = "channel",
        combine_data_arrays: bool = False,
    ):
        """
        Select channels

        Args:
            source_datapipe: Datapipe of Xarray objects
            channels: Channel names to keep
            dim_name: Dimension name for time
            combine_data_arrays: Combine the data arrays into a single DataArray after selecting variables
                (Useful for NWPs where each variable is its own data variable)
        """
        self.source_datapipe = source_datapipe
        self.channels = channels
        self.dim_name = dim_name
        self.combine_data_arrays = combine_data_arrays

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        for xr_data in self.source_datapipe:
            if self.combine_data_arrays:
                # Drop data variables whose names are not in channels
                xr_data = xr_data.drop_vars(
                    [v for v in xr_data.data_vars if v not in self.channels]
                )
                xr_data = xr_data.to_array(dim=self.dim_name)
            logger.debug(f"Selecting Channels: {self.channels} out of {xr_data['channel'].values}")
            xr_data = xr_data.sel({self.dim_name: list(self.channels)})
            yield xr_data
