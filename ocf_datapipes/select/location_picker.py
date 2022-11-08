"""Pick locations from a dataset"""
import logging
from typing import Optional

import numpy as np
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.utils.consts import Location

logger = logging.getLogger(__name__)


@functional_datapipe("location_picker")
class LocationPickerIterDataPipe(IterDataPipe):
    """Picks locations from a dataset and returns them"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        return_all_locations: bool = False,
        x_dim_name: Optional[str] = "x_osgb",
        y_dim_name: Optional[str] = "y_osgb",
    ):
        """
        Picks locations from a dataset and returns them

        Args:
            source_datapipe: Datapipe emitting Xarray Dataset
            return_all_locations: Whether to return all locations,
                if True, also returns them in order
            x_dim_name: x dimension name, defaulted to 'x_osgb'
            y_dim_name: y dimension name, defaulted to 'y_osgb'
        """
        super().__init__()
        self.source_datapipe = source_datapipe
        self.return_all_locations = return_all_locations
        self.x_dim_name = x_dim_name
        self.y_dim_name = y_dim_name

    def __iter__(self) -> Location:
        """Returns locations from the inputs datapipe"""
        for xr_dataset in self.source_datapipe:

            logger.debug(f"Getting locations for {xr_dataset}")

            if self.return_all_locations:
                # Iterate through all locations in dataset
                for location_idx in range(len(xr_dataset[self.x_dim_name])):
                    location = Location(
                        x=xr_dataset[self.x_dim_name][location_idx].values,
                        y=xr_dataset[self.y_dim_name][location_idx].values,
                    )
                    if "pv_system_id" in xr_dataset.coords.keys():

                        location.id = int(xr_dataset["pv_system_id"][location_idx].values)
                    logger.debug(f"Got all locations {location}")
                    yield location
            else:
                # Assumes all datasets have osgb coordinates for selecting locations
                # Pick 1 random location from the input dataset
                location_idx = np.random.randint(0, len(xr_dataset[self.x_dim_name]))
                location = Location(
                    x=xr_dataset[self.x_dim_name][location_idx].values,
                    y=xr_dataset[self.y_dim_name][location_idx].values,
                )
                if "pv_system_id" in xr_dataset.coords.keys():
                    location.id = int(xr_dataset["pv_system_id"][location_idx].values)
                    logger.debug(f"Have selected location.id {location.id}")
                for i in range(0, 10):
                    yield location
