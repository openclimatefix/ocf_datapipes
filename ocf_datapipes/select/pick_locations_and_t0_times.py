"""Select the t0 time and lcoation for training"""
import logging
from typing import Optional

import numpy as np
import pandas as pd
from torch.utils.data import IterDataPipe, functional_datapipe

from ocf_datapipes.utils.consts import Location
from ocf_datapipes.utils.geospatial import (
    spatial_coord_type,
)

logger = logging.getLogger(__name__)


@functional_datapipe("select_loc_and_t0")
class LocationT0PickerIterDataPipe(IterDataPipe):
    """Datapipe to yield location-time pairs from the input data source."""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        return_all: bool = False,
        shuffle: bool = False,
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
            time_dim_name: time dimension name, defaulted to 'time_utc'
        """
        super().__init__()
        self.source_datapipe = source_datapipe
        self.return_all = return_all
        self.shuffle = shuffle
        self.time_dim_name = time_dim_name

    def _yield_all_iter(self, xr_dataset):
        xr_coord_system, xr_x_dim, xr_y_dim = spatial_coord_type(xr_dataset)
        t_index, x_index = np.meshgrid(
            np.arange(len(xr_dataset[self.time_dim_name])),
            np.arange(len(xr_dataset[xr_x_dim])),
        )

        index_pairs = np.stack((t_index.ravel(), x_index.ravel())).T

        if self.shuffle:
            index_pairs = np.random.permutation(index_pairs)

        # Iterate through all locations in dataset
        for t_index, loc_index in index_pairs:
            t0 = xr_dataset[self.time_dim_name][t_index].values
            location = Location(
                coordinate_system=xr_coord_system,
                x=xr_dataset[xr_x_dim][loc_index].values,
                y=xr_dataset[xr_y_dim][loc_index].values,
            )

            # for pv
            if "pv_system_id" in xr_dataset.coords.keys():
                location.id = int(xr_dataset["pv_system_id"][loc_index].values)

            # for gsp
            if "gsp_id" in xr_dataset.coords.keys():
                location.id = int(xr_dataset["gsp_id"][loc_index].values)

            # for sensor
            if "station_id" in xr_dataset.coords.keys():
                location.id = int(xr_dataset["station_id"][loc_index].values)

            yield location, t0

    def _yield_random_iter(self, xr_dataset):
        xr_coord_system, xr_x_dim, xr_y_dim = spatial_coord_type(xr_dataset)
        while True:
            location_idx = np.random.randint(0, len(xr_dataset[xr_x_dim]))

            location = Location(
                coordinate_system=xr_coord_system,
                x=xr_dataset[xr_x_dim][location_idx].values,
                y=xr_dataset[xr_y_dim][location_idx].values,
            )
            if "pv_system_id" in xr_dataset.coords.keys():
                location.id = int(xr_dataset["pv_system_id"][location_idx].values)

            # for gsp
            if "gsp_id" in xr_dataset.coords.keys():
                location.id = int(xr_dataset["gsp_id"][location_idx].values)

            # for sensor
            if "station_id" in xr_dataset.coords.keys():
                location.id = int(xr_dataset["station_id"][location_idx].values)

            t0 = np.random.choice(xr_dataset[self.time_dim_name].values)

            yield location, t0

    def __iter__(self) -> tuple[Location, pd.Timestamp]:
        xr_dataset = next(iter(self.source_datapipe))

        if self.return_all:
            return self._yield_all_iter(xr_dataset)
        else:
            return self._yield_random_iter(xr_dataset)
