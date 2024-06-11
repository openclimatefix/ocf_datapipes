"""Pick locations from a dataset"""

import logging

import numpy as np
from torch.utils.data import IterDataPipe, functional_datapipe

from ocf_datapipes.utils import Location
from ocf_datapipes.utils.geospatial import spatial_coord_type

logger = logging.getLogger(__name__)


@functional_datapipe("pick_locations")
class PickLocationsIterDataPipe(IterDataPipe):
    """Picks random locations from a dataset"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        return_all: bool = False,
        shuffle: bool = False,
    ):
        """
        Datapipe to yield locations from the input data source.

        Args:
            source_datapipe: Datapipe emitting Xarray Dataset
            return_all: Whether to return all t0-location pairs,
                if True, also returns them in structured order
            shuffle: If `return_all` is True this sets whether the pairs are
                shuffled before being returned.
        """
        super().__init__()
        self.source_datapipe = source_datapipe
        self.return_all = return_all
        self.shuffle = shuffle

    def _yield_all_iter(self, xr_dataset):
        """Samples without replacement from possible locations"""
        # Get the spatial coords
        xr_coord_system, xr_x_dim, xr_y_dim = spatial_coord_type(xr_dataset)

        loc_indices = np.arange(len(xr_dataset[xr_x_dim]))

        if self.shuffle:
            loc_indices = np.random.permutation(loc_indices)

        # Iterate through all locations in dataset
        for loc_index in loc_indices:
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

            yield location

    def _yield_random_iter(self, xr_dataset):
        """Samples with replacement from possible locations"""
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

            yield location

    def __iter__(self) -> Location:
        xr_dataset = next(iter(self.source_datapipe))

        if self.return_all:
            return self._yield_all_iter(xr_dataset)
        else:
            return self._yield_random_iter(xr_dataset)
