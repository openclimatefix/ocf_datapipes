"""Select the t0 time for training"""
import logging

import numpy as np
import pandas as pd
from torch.utils.data import IterDataPipe, functional_datapipe

logger = logging.getLogger(__name__)


@functional_datapipe("pick_t0_times")
class PickT0TimesIterDataPipe(IterDataPipe):
    """Picks random t0 times from a dataset"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        dim_name: str = "time_utc",
    ):
        """
        Picks random t0 times from a dataset

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
                raise Exception("There are no values to get t0 from")
            t0 = np.random.choice(xr_data[self.dim_name].values)
            logger.debug(f"t0 will be {t0}")

            yield t0
