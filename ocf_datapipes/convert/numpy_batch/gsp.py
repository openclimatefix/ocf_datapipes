"""Convert GSP to Numpy Batch"""

import logging

from torch.utils.data import IterDataPipe, functional_datapipe

from ocf_datapipes.batch import BatchKey, NumpyBatch
from ocf_datapipes.utils.utils import datetime64_to_float

logger = logging.getLogger(__name__)


def convert_gsp_to_numpy_batch(xr_data):
    """Convert from Xarray to NumpyBatch"""

    example: NumpyBatch = {
        BatchKey.gsp: xr_data.values,
        BatchKey.gsp_t0_idx: xr_data.attrs["t0_idx"],
        BatchKey.gsp_id: xr_data.gsp_id.values,
        BatchKey.gsp_nominal_capacity_mwp: xr_data.isel(time_utc=0)["nominal_capacity_mwp"].values,
        BatchKey.gsp_effective_capacity_mwp: (
            xr_data.isel(time_utc=0)["effective_capacity_mwp"].values
        ),
        BatchKey.gsp_time_utc: datetime64_to_float(xr_data["time_utc"].values),
    }

    # Coordinates
    for batch_key, dataset_key in (
        (BatchKey.gsp_y_osgb, "y_osgb"),
        (BatchKey.gsp_x_osgb, "x_osgb"),
    ):
        if dataset_key in xr_data.coords.keys():
            example[batch_key] = xr_data[dataset_key].values

    return example


@functional_datapipe("convert_gsp_to_numpy_batch")
class ConvertGSPToNumpyBatchIterDataPipe(IterDataPipe):
    """Convert GSP Xarray to NumpyBatch"""

    def __init__(self, source_datapipe: IterDataPipe):
        """
        Convert GSP Xarray to NumpyBatch object

        Args:
            source_datapipe: Datapipe emitting GSP Xarray object
        """
        super().__init__()
        self.source_datapipe = source_datapipe

    def __iter__(self) -> NumpyBatch:
        """Convert from Xarray to NumpyBatch"""
        for xr_data in self.source_datapipe:
            yield convert_gsp_to_numpy_batch(xr_data)
