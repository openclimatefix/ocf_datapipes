"""Selects time slice"""
from typing import Union

import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe, Zipper


@functional_datapipe("select_id")
class SelectIDIterDataPipe(IterDataPipe):
    """Selects time slice"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        location_datapipe: IterDataPipe,
    ):
        """
        Selects id

        Args:
            source_datapipe: Datapipe of Xarray objects
            location_datapipe: Location datapipe
        """
        self.source_datapipe = source_datapipe
        self.location_datapipe = location_datapipe

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        for xr_data, location in Zipper(self.source_datapipe, self.location_datapipe):

            xr_data = xr_data.sel(id=location.id)
            yield xr_data
