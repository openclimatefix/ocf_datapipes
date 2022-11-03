"""Drop GSP output from xarray"""
from typing import List

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@functional_datapipe("drop_gsp")
class DropGSPIterDataPipe(IterDataPipe):
    """Drops GSP"""

    def __init__(self, source_datapipe: IterDataPipe, gsps_to_keep: List[int] = slice(1, 317)):
        """
        Drop GSP National from the dataarray

        Args:
            source_datapipe: Datapipe emitting GSP xarray
            gsps_to_keep: List of GSP Id's to keep, by default all the 317 non-national ones
        """
        self.source_datapipe = source_datapipe
        self.gsps_to_keep = gsps_to_keep

    def __iter__(self):
        for xr_data in self.source_datapipe:
            # GSP ID 0 is national
            xr_data = xr_data.isel(gsp_id=self.gsps_to_keep)
            yield xr_data
