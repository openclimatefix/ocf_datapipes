"""Pick locations from a dataset"""
import logging

import numpy as np
from torch.utils.data import IterDataPipe, functional_datapipe

from ocf_datapipes.utils.consts import Location
from ocf_datapipes.utils.geospatial import spatial_coord_type

logger = logging.getLogger(__name__)


@functional_datapipe("location_picker")
class LocationPickerIterDataPipe(IterDataPipe):
    """Picks locations from a dataset and returns them"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        return_all_locations: bool = False,
    ):
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
            loc_type, xr_x_dim, xr_y_dim = spatial_coord_type(xr_dataset)

            if self.return_all_locations:
                logger.debug("Going to return all locations")

                # Iterate through all locations in dataset
                for location_idx in range(len(xr_dataset[xr_x_dim])):
                    location = Location(
                        x=xr_dataset[xr_x_dim][location_idx].values,
                        y=xr_dataset[xr_y_dim][location_idx].values,
                        coordinate_system=loc_type,
                    )
                    if "pv_system_id" in xr_dataset.coords.keys():
                        location.id = int(xr_dataset["pv_system_id"][location_idx].values)
                    logger.debug(f"Got all location {location}")
                    yield location
            else:
                # Pick 1 random location from the input dataset
                logger.debug("Selecting random idx")
                location_idx = np.random.randint(0, len(xr_dataset[xr_x_dim]))
                logger.debug(f"{location_idx=}")
                location = Location(
                    x=xr_dataset[xr_x_dim][location_idx].values,
                    y=xr_dataset[xr_y_dim][location_idx].values,
                    coordinate_system=loc_type,
                )
                if "pv_system_id" in xr_dataset.coords.keys():
                    location.id = int(xr_dataset["pv_system_id"][location_idx].values)
                    logger.debug(f"Have selected location.id {location.id}")
                logger.debug(f"{location=}")
                yield location
