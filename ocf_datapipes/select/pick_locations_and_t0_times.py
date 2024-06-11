"""Select the t0 time and lcoation for training"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from torch.utils.data import IterDataPipe, functional_datapipe

from ocf_datapipes.utils import Location
from ocf_datapipes.utils.geospatial import (
    spatial_coord_type,
)

logger = logging.getLogger(__name__)


@functional_datapipe("pick_locs_and_t0s")
class PickLocationsAndT0sIterDataPipe(IterDataPipe):
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
            shuffle: If `return_all` is True this sets whether the pairs are
                shuffled before being returned.
            time_dim_name: time dimension name, defaulted to 'time_utc'
        """
        super().__init__()
        self.source_datapipe = source_datapipe
        self.return_all = return_all
        self.shuffle = shuffle
        self.time_dim_name = time_dim_name

    def _yield_all_iter(self, xr_dataset):
        # Get the spatial coords
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
            # Get the location ID
            loc_id = None
            for id_dim_name in ["pv_system_id", "gsp_id", "station_id"]:
                if id_dim_name in xr_dataset.coords.keys():
                    loc_id = int(xr_dataset[id_dim_name][loc_index].values)

            t0 = xr_dataset[self.time_dim_name][t_index].values
            location = Location(
                coordinate_system=xr_coord_system,
                x=xr_dataset[xr_x_dim][loc_index].values,
                y=xr_dataset[xr_y_dim][loc_index].values,
                id=loc_id,
            )

            yield location, t0

    def _yield_random_iter(self, xr_dataset):
        # Get the spatial coords
        xr_coord_system, xr_x_dim, xr_y_dim = spatial_coord_type(xr_dataset)

        while True:
            loc_index = np.random.randint(0, len(xr_dataset[xr_x_dim]))

            # Get the location ID
            loc_id = None
            for id_dim_name in ["pv_system_id", "gsp_id", "station_id"]:
                if id_dim_name in xr_dataset.coords.keys():
                    loc_id = int(xr_dataset[id_dim_name][loc_index].values)

            location = Location(
                coordinate_system=xr_coord_system,
                x=xr_dataset[xr_x_dim][loc_index].values,
                y=xr_dataset[xr_y_dim][loc_index].values,
                id=loc_id,
            )

            t0 = np.random.choice(xr_dataset[self.time_dim_name].values)

            yield location, t0

    def __iter__(self) -> tuple[Location, pd.Timestamp]:
        xr_dataset = next(iter(self.source_datapipe))

        if self.return_all:
            return self._yield_all_iter(xr_dataset)
        else:
            return self._yield_random_iter(xr_dataset)
