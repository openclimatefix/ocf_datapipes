"""Convert PV to Numpy Batch"""

import logging

import numpy as np
from torch.utils.data import IterDataPipe, functional_datapipe

from ocf_datapipes.batch import BatchKey, NumpyBatch
from ocf_datapipes.utils.utils import datetime64_to_float

logger = logging.getLogger(__name__)


def convert_pv_to_numpy_batch(xr_data):
    """Convert PV Xarray to NumpyBatch"""
    example: NumpyBatch = {
        BatchKey.pv: xr_data.values,
        BatchKey.pv_t0_idx: xr_data.attrs["t0_idx"],
        BatchKey.pv_ml_id: xr_data["ml_id"].values,
        BatchKey.pv_id: xr_data["pv_system_id"].values.astype(np.float32),
        BatchKey.pv_observed_capacity_wp: (xr_data["observed_capacity_wp"].values),
        BatchKey.pv_nominal_capacity_wp: (xr_data["nominal_capacity_wp"].values),
        BatchKey.pv_time_utc: datetime64_to_float(xr_data["time_utc"].values),
        BatchKey.pv_latitude: xr_data["latitude"].values,
        BatchKey.pv_longitude: xr_data["longitude"].values,
    }

    return example


@functional_datapipe("convert_pv_to_numpy_batch")
class ConvertPVToNumpyBatchIterDataPipe(IterDataPipe):
    """Convert PV Xarray to NumpyBatch"""

    def __init__(self, source_datapipe: IterDataPipe):
        """
        Convert PV Xarray objects to NumpyBatch objects

        Args:
            source_datapipe: Datapipe emitting PV Xarray objects
        """
        super().__init__()
        self.source_datapipe = source_datapipe

    def __iter__(self) -> NumpyBatch:
        """Convert PV Xarray to NumpyBatch"""
        for xr_data in self.source_datapipe:
            yield convert_pv_to_numpy_batch(xr_data)
