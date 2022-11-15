"""Select the t0 time for training"""
import logging
from typing import Optional

import numpy as np
import pandas as pd
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

logger = logging.getLogger(__name__)


@functional_datapipe("select_t0_time")
class SelectT0TimeIterDataPipe(IterDataPipe):
    """Select the random t0 time for the training data"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        dim_name: str = "time_utc",
        return_all_times: Optional[bool] = False,
        number_locations_datapipe: Optional[IterDataPipe] = None,
    ):
        """
        Select a random t0 time for training

        Args:
            source_datapipe: Datapipe emitting Xarray objects
            dim_name: The time dimension name to use
            return_all_times: option to
            number_locations_datapipe: get the total number of locations
        """
        self.source_datapipe = source_datapipe
        self.dim_name = dim_name
        self.return_all_times = return_all_times
        self.number_locations_datapipe = number_locations_datapipe

        if self.return_all_times:
            logger.debug("Will be returning all t0 times")

    def __iter__(self) -> pd.Timestamp:
        """Get the latest timestamp and return it"""
        for xr_data in self.source_datapipe:

            if self.return_all_times and (self.number_locations_datapipe is not None):

                number_of_locations = next(self.number_locations_datapipe)

                logger.info(
                    f"Will be returning all times from {xr_data[self.dim_name]}. "
                    f"There are {len(xr_data[self.dim_name])} of them"
                )

                for t0 in xr_data[self.dim_name].values:

                    for _ in range(0, number_of_locations):
                        logger.debug(f"t0 will be {t0}")
                        yield t0

            else:

                logger.debug(f"Selecting t0 from {len(xr_data[self.dim_name])} datetimes")

                if len(xr_data[self.dim_name].values) == 0:
                    raise Exception("There are no values to get t0 from")
                t0 = np.random.choice(xr_data[self.dim_name].values)
                logger.debug(f"t0 will be {t0}")

                yield t0
