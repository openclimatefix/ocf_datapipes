import numpy as np
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.utils.consts import BatchKey, NumpyBatch
from ocf_datapipes.utils.utils import datetime64_to_float


@functional_datapipe("convert_pv_to_numpy_batch")
class ConvertPVToNumpyBatchIterDataPipe(IterDataPipe):
    def __init__(self, source_dp: IterDataPipe):
        super().__init__()
        self.source_dp = source_dp

    def __iter__(self) -> NumpyBatch:
        for xr_data in self.source_dp:
            example: NumpyBatch = {
                BatchKey.pv: xr_data.values,
                BatchKey.pv_t0_idx: xr_data.attrs["t0_idx"],
                BatchKey.pv_system_row_number: xr_data["pv_system_row_number"].values,
                BatchKey.pv_id: xr_data["pv_system_id"].values.astype(np.float32),
                BatchKey.pv_capacity_wp: xr_data["capacity_wp"].values,
                BatchKey.pv_time_utc: datetime64_to_float(xr_data["time_utc"].values),
                BatchKey.pv_x_osgb: xr_data["x_osgb"].values,
                BatchKey.pv_y_osgb: xr_data["y_osgb"].values,
            }

            yield example
