"""Convert PV to Numpy Batch"""

import logging

import numpy as np
from torch.utils.data import IterDataPipe, functional_datapipe

from ocf_datapipes.batch import BatchKey, NumpyBatch
from ocf_datapipes.utils.utils import datetime64_to_float

logger = logging.getLogger(__name__)


def convert_sensor_to_numpy_batch(xr_data):
    """Convert Sensor Xarray to NumpyBatch"""

    example: NumpyBatch = {
        BatchKey.sensor: xr_data.values,
        BatchKey.sensor_t0_idx: xr_data.attrs["t0_idx"],
        BatchKey.sensor_id: xr_data["station_id"].values.astype(np.float32),
        # BatchKey.sensor_observed_capacity_wp: (xr_data["observed_capacity_wp"].values),
        # BatchKey.sensor_nominal_capacity_wp: (xr_data["nominal_capacity_wp"].values),
        BatchKey.sensor_time_utc: datetime64_to_float(xr_data["time_utc"].values),
        BatchKey.sensor_latitude: xr_data["latitude"].values,
        BatchKey.sensor_longitude: xr_data["longitude"].values,
    }
    return example


@functional_datapipe("convert_sensor_to_numpy_batch")
class ConvertSensorToNumpyBatchIterDataPipe(IterDataPipe):
    """Convert Sensor Xarray to NumpyBatch"""

    def __init__(self, source_datapipe: IterDataPipe):
        """
        Convert sensor Xarray objects to NumpyBatch objects

        Args:
            source_datapipe: Datapipe emitting sensor Xarray objects
        """
        super().__init__()
        self.source_datapipe = source_datapipe

    def __iter__(self) -> NumpyBatch:
        """Iterate and convert sensor Xarray to NumpyBatch"""
        for xr_data in self.source_datapipe:
            yield convert_sensor_to_numpy_batch(xr_data)
