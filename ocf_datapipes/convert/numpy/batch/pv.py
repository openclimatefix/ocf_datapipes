"""Convert PV to Numpy Batch"""
import logging

import numpy as np
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.utils.consts import BatchKey, NumpyBatch
from ocf_datapipes.utils.utils import datetime64_to_float

logger = logging.getLogger(__name__)


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
        """Iterate and convert PV Xarray to NumpyBatch"""
        for xr_data in self.source_datapipe:

            logger.debug("Converting PV xarray to numpy example")

            example: NumpyBatch = {
                BatchKey.pv: xr_data.values,
                BatchKey.pv_t0_idx: xr_data.attrs["t0_idx"],
                BatchKey.pv_system_row_number: xr_data["pv_system_row_number"].values,
                BatchKey.pv_id: xr_data["pv_system_id"].values.astype(np.float32),
                BatchKey.pv_capacity_watt_power: xr_data["capacity_watt_power"].values,
                BatchKey.pv_time_utc: datetime64_to_float(xr_data["time_utc"].values),
                BatchKey.pv_x_osgb: xr_data["x_osgb"].values,
                BatchKey.pv_y_osgb: xr_data["y_osgb"].values,
            }

            yield example
