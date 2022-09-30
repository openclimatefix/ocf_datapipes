"""Select the t0 time for training"""
import logging

import numpy as np
import pandas as pd
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

logger = logging.getLogger(__name__)


@functional_datapipe("select_t0_time")
class SelectT0TimeIterDataPipe(IterDataPipe):
    """Select the random t0 time for the training data"""

    def __init__(self, source_datapipe: IterDataPipe, dim_name: str = "time_utc"):
        """
        Select a random t0 time for training

        Args:
            source_datapipe: Datapipe emitting Xarray objects
            dim_name: The time dimension name to use
        """
        self.source_datapipe = source_datapipe
        self.dim_name = dim_name

    def __iter__(self) -> pd.Timestamp:
        """Get the latest timestamp and return it"""
        for xr_data in self.source_datapipe:

            logger.debug(f"Selecting t0 from {len(xr_data[self.dim_name])} datetimes")

            if len(xr_data[self.dim_name].values) == 0:
                assert Exception("There are no values to get t0 from")
            t0 = np.random.choice(xr_data[self.dim_name].values)
            logger.debug(f"t0 will be {t0}")

            yield t0
