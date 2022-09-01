"""Pick locations from a dataset"""
import logging

import numpy as np
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.utils.consts import Location

logger = logging.getLogger(__name__)


@functional_datapipe("location_picker")
class LocationPickerIterDataPipe(IterDataPipe):
    """Picks locations from a dataset and returns them"""

    def __init__(self, source_datapipe: IterDataPipe, return_all_locations: bool = False):
        """
        Picks locations from a dataset and returns them

        Args:
            source_datapipe: Datapipe emitting Xarray Dataset
            return_all_locations: Whether to return all locations,
            if True, also returns them in order
        """
        super().__init__()
        self.source_datapipe = source_datapipe
        self.return_all_locations = return_all_locations

    def __iter__(self) -> Location:
        """Returns locations from the inputs datapipe"""
        for xr_dataset in self.source_datapipe:

            logger.debug(f"Getting locations for {xr_dataset}")

            if self.return_all_locations:
                # Iterate through all locations in dataset
                for location_idx in range(len(xr_dataset["x_osgb"])):
                    location = Location(
                        x=xr_dataset["x_osgb"][location_idx].values,
                        y=xr_dataset["y_osgb"][location_idx].values,
                    )
                    logger.debug(f"Got all locations {location}")
                    yield location
            else:
                # Assumes all datasets have osgb coordinates for selecting locations
                # Pick 1 random location from the input dataset
                location_idx = np.random.randint(0, len(xr_dataset["x_osgb"]))
                location = Location(
                    x=xr_dataset["x_osgb"][location_idx].values,
                    y=xr_dataset["y_osgb"][location_idx].values,
                )
                yield location
