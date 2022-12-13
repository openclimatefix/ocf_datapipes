"""Convert point PV sites to image output"""
import logging
from typing import Optional

import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

logger = logging.getLogger(__name__)


@functional_datapipe("remove_northern_gsp")
class RemoveNorthernGSPIterDataPipe(IterDataPipe):
    """Remove northern GSPs"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        northern_y_osgb_limit: Optional[int] = 1_036_975,
    ):
        """
        Remove northern GSP. This might be because satellite data is not available

        Args:
            source_datapipe: Source datapipe of GSP data
            northern_y_osgb_limit: limit all gsp above this limit.
                The deafult gets rid of the very northern ones
        """
        self.source_datapipe = source_datapipe
        self.northern_y_osgb_limit = northern_y_osgb_limit

    def __iter__(self) -> xr.DataArray:
        for source_datapipe in self.source_datapipe:

            logger.debug(
                f"Removing any gsp with Y OSGB greater than {self.northern_y_osgb_limit}. "
                f"There are currently {len(source_datapipe.gsp_id)} GSPs"
            )

            keep_index = source_datapipe.y_osgb < self.northern_y_osgb_limit
            keep_gsp_ids = source_datapipe.gsp_id[keep_index]

            source_datapipe = source_datapipe.sel(gsp_id=keep_gsp_ids)
            logger.debug(f"There are now {len(source_datapipe.gsp_id)} GSPs")

            yield source_datapipe
