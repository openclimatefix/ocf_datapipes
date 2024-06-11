"""Convert Wind to Numpy Batch"""

import logging

import numpy as np
from torch.utils.data import IterDataPipe, functional_datapipe

from ocf_datapipes.batch import BatchKey, NumpyBatch
from ocf_datapipes.utils.utils import datetime64_to_float

logger = logging.getLogger(__name__)


def convert_wind_to_numpy_batch(xr_data):
    """Convert Wind Xarray to NumpyBatch"""

    example: NumpyBatch = {
        BatchKey.wind: xr_data.values,
        BatchKey.wind_t0_idx: xr_data.attrs["t0_idx"],
        BatchKey.wind_ml_id: xr_data["ml_id"].values,
        BatchKey.wind_id: xr_data["wind_system_id"].values.astype(np.float32),
        BatchKey.wind_observed_capacity_mwp: (xr_data["observed_capacity_mwp"].values),
        BatchKey.wind_nominal_capacity_mwp: (xr_data["nominal_capacity_mwp"].values),
        BatchKey.wind_time_utc: datetime64_to_float(xr_data["time_utc"].values),
        BatchKey.wind_latitude: xr_data["latitude"].values,
        BatchKey.wind_longitude: xr_data["longitude"].values,
    }

    return example


@functional_datapipe("convert_wind_to_numpy_batch")
class ConvertWindToNumpyBatchIterDataPipe(IterDataPipe):
    """Convert Wind Xarray to NumpyBatch"""

    def __init__(self, source_datapipe: IterDataPipe):
        """
        Convert Wind Xarray objects to NumpyBatch objects

        Args:
            source_datapipe: Datapipe emitting Wind Xarray objects
        """
        super().__init__()
        self.source_datapipe = source_datapipe

    def __iter__(self) -> NumpyBatch:
        """Iterate and convert PV Xarray to NumpyBatch"""
        for xr_data in self.source_datapipe:
            yield convert_wind_to_numpy_batch(xr_data)
