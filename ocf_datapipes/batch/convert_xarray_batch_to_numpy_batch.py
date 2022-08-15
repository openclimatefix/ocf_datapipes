from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
import xarray as xr
import numpy as np
from ocf_datapipes.utils import NumpyBatch

@functional_datapipe("convert_xarray_batch_to_numpy_batch")
class ConvertXarrayBatchToNumpyBatch(IterDataPipe):
    def __init__(self, source_dp: IterDataPipe):
        self.source_dp = source_dp

    def __iter__(self) -> NumpyBatch:
        for xr_batch in self.source_dp:
            # TODO Actually convert it, or add conversion to XarrayBatch
            np_batch: NumpyBatch = xr_batch.to_numpy()
            yield np_batch
