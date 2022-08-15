from typing import Iterable, Union

import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe, Zipper

from ocf_datapipes.utils import XarrayBatch


@functional_datapipe("convert_to_xarray_batch")
class ConvertToXarrayBatchIterDataPipe(IterDataPipe):
    def __init__(self, source_dps: Iterable[IterDataPipe]):
        super().__init__()
        self.source_dps = source_dps

    def __iter__(self) -> XarrayBatch:
        for xr_datas in Zipper(*self.source_dps):
            xr_batch = XarrayBatch()
            # TODO Combine here to batch
            yield xr_batch
