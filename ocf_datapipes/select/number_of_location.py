"""Get the numner of locations from a dataset"""
import logging
from typing import Optional

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.utils.consts import Location

logger = logging.getLogger(__name__)


@functional_datapipe("number_of_locations")
class NumberOfLocationsrIterDataPipe(IterDataPipe):
    """Picks locations from a dataset and returns them"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        x_dim_name: Optional[str] = "x_osgb",
    ):
        """
        Get the total number locations from a dataset and returns them

        Args:
            source_datapipe: Datapipe emitting Xarray Dataset
                if True, also returns them in order
            x_dim_name: x dimension name, defaulted to 'x_osgb'
        """
        super().__init__()
        self.source_datapipe = source_datapipe
        self.x_dim_name = x_dim_name

    def __iter__(self) -> Location:
        """Returns the nuber of locations from the inputs datapipe"""
        for xr_dataset in self.source_datapipe:
            yield len(xr_dataset[self.x_dim_name])
