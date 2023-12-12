"""Convert PV to Numpy Batch"""
import logging

import numpy as np
from torch.utils.data import IterDataPipe, functional_datapipe

from ocf_datapipes.utils.consts import BatchKey, NumpyBatch
from ocf_datapipes.utils.utils import datetime64_to_float

logger = logging.getLogger(__name__)


@functional_datapipe("convert_sensor_to_numpy_batch")
class ConvertSensorToNumpyBatchIterDataPipe(IterDataPipe):
    """Convert Sensor Xarray to NumpyBatch"""

    def __init__(self, source_datapipe: IterDataPipe):
        """
        Convert PV Xarray objects to NumpyBatch objects

        Args:
            source_datapipe: Datapipe emitting PV Xarray objects
        """
        super().__init__()
        self.source_datapipe = source_datapipe

    def __iter__(self) -> NumpyBatch:
        """Iterate and convert PV Xarray to NumpyBatch"""
        for xr_data in self.source_datapipe:
            logger.debug("Converting Sensor xarray to numpy example")

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

            yield example
