from torchdata.datapipes.iter import IterDataPipe, Zipper
from torchdata.datapipes import functional_datapipe
from ocf_datapipes.utils.consts import NumpyBatch, BatchKey


@functional_datapipe("merge_numpy_modalities")
class MergeNumpyModalitiesIterDataPipe(IterDataPipe):
    def __init__(self, source_datapipes: [IterDataPipe]):
        self.source_datapipes = source_datapipes

    def __iter__(self) -> NumpyBatch:
        for np_batches in Zipper(*self.source_datapipes):
            example: NumpyBatch = {}
            for np_batch in np_batches:
                example.update(np_batch)
            yield example
