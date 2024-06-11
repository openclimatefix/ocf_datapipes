"""Select the t0 time for training"""

import logging

import numpy as np
import pandas as pd
from torch.utils.data import IterDataPipe, functional_datapipe

logger = logging.getLogger(__name__)


@functional_datapipe("pick_t0_times")
class PickT0TimesIterDataPipe(IterDataPipe):
    """Picks (random) t0 times from a dataset"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        return_all: bool = False,
        shuffle: bool = False,
        dim_name: str = "time_utc",
    ):
        """
        Datapipe to yield t0 times from the input data source.

        Args:
            source_datapipe: Datapipe emitting Xarray objects.
            return_all: Whether to return all t0 values, else sample with replacement. If True, the
                default behaviour to return t0 values in order - see `shuffle` parameter.
            shuffle: If `return_all` is True this sets whether the pairs are
                shuffled before being returned.
            dim_name: The time dimension name to use.
        """
        self.source_datapipe = source_datapipe
        self.return_all = return_all
        self.shuffle = shuffle
        self.dim_name = dim_name

    def _yield_random_iter(self, xr_dataset):
        """Sample t0 with replacement"""
        while True:
            t0 = np.random.choice(xr_dataset[self.dim_name].values)
            yield t0

    def _yield_all_iter(self, xr_dataset):
        """Yield all the t0s in order, and maybe with a shuffle"""
        all_t0s = np.copy(xr_dataset[self.dim_name].values)
        if self.shuffle:
            all_t0s = np.random.permutation(all_t0s)
        for t0 in all_t0s:
            yield t0

    def __iter__(self) -> pd.Timestamp:
        xr_dataset = next(iter(self.source_datapipe))

        if len(xr_dataset[self.dim_name].values) == 0:
            raise Exception("There are no values to get t0 from")

        if self.return_all:
            return self._yield_all_iter(xr_dataset)
        else:
            return self._yield_random_iter(xr_dataset)
