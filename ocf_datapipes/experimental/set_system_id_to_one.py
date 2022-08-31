import numpy as np
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.utils.consts import BatchKey, NumpyBatch


@functional_datapipe("set_system_ids_to_one")
class SetSystemIDsToOneIterDataPipe(IterDataPipe):
    def __init__(self, source_datapipe: IterDataPipe):
        self.source_datapipe = source_datapipe

    def __iter__(self) -> NumpyBatch:
        for np_batch in self.source_datapipe:
            np_batch[BatchKey.gsp_id] = np.ones_like(np_batch[BatchKey.gsp_id])
            np_batch[BatchKey.pv_id] = np.ones_like(np_batch[BatchKey.pv_id])
            yield np_batch
