"""Select the t0 time and lcoation for training"""
import logging
from typing import Optional

import numpy as np
import pandas as pd
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.utils.consts import Location

logger = logging.getLogger(__name__)


@functional_datapipe("select_loc_and_t0")
class LocationT0PickerIterDataPipe(IterDataPipe):
    """Datapipe to yield location-time pairs from the input data source."""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        return_all: bool = False,
        shuffle: bool = False,
        x_dim_name: Optional[str] = "x_osgb",
        y_dim_name: Optional[str] = "y_osgb",
        time_dim_name: Optional[str] = "time_utc",
    ):
        """
        Datapipe to yield location-time pairs from the input data source.

        Args:
            source_datapipe: Datapipe emitting Xarray Dataset
            return_all: Whether to return all t0-location pairs,
                if True, also returns them in structured order
            shuffle: If `return_all` sets whether the pairs are
                shuffled before being returned.
            x_dim_name: x dimension name, defaulted to 'x_osgb'
            y_dim_name: y dimension name, defaulted to 'y_osgb'
            time_dim_name: time dimension name, defaulted to 'time_utc'
        """
        super().__init__()
        self.source_datapipe = source_datapipe
        self.return_all = return_all
        self.shuffle = shuffle
        self.x_dim_name = x_dim_name
        self.y_dim_name = y_dim_name
        self.time_dim_name = time_dim_name

    def _yield_all_iter(self, xr_dataset):
        t_index, x_index = np.meshgrid(
            np.arange(len(xr_dataset[self.time_dim_name])),
            np.arange(len(xr_dataset[self.x_dim_name])),
        )

        index_pairs = np.stack((t_index.ravel(), x_index.ravel())).T

        if self.shuffle:
            index_pairs = np.random.permutation(index_pairs)

        # Iterate through all locations in dataset
        for t_index, loc_index in index_pairs:
            t0 = xr_dataset[self.time_dim_name][t_index].values
            location = Location(
                x=xr_dataset[self.x_dim_name][loc_index].values,
                y=xr_dataset[self.y_dim_name][loc_index].values,
            )

            # for pv
            if "pv_system_id" in xr_dataset.coords.keys():
                location.id = int(xr_dataset["pv_system_id"][loc_index].values)

            # for gsp
            if "gsp_id" in xr_dataset.coords.keys():
                location.id = int(xr_dataset["gsp_id"][loc_index].values)

            yield location, t0

    def _yield_random_iter(self, xr_dataset):
        while True:
            location_idx = np.random.randint(0, len(xr_dataset[self.x_dim_name]))

            location = Location(
                x=xr_dataset[self.x_dim_name][location_idx].values,
                y=xr_dataset[self.y_dim_name][location_idx].values,
            )
            if "pv_system_id" in xr_dataset.coords.keys():
                location.id = int(xr_dataset["pv_system_id"][location_idx].values)

            # for gsp
            if "gsp_id" in xr_dataset.coords.keys():
                location.id = int(xr_dataset["gsp_id"][location_idx].values)

            t0 = np.random.choice(xr_dataset[self.time_dim_name].values)

            yield location, t0

    def __iter__(self) -> tuple[Location, pd.Timestamp]:
        xr_dataset = next(iter(self.source_datapipe))

        if self.return_all:
            return self._yield_all_iter(xr_dataset)
        else:
            return self._yield_random_iter(xr_dataset)
