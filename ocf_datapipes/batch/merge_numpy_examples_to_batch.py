from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.utils.consts import NumpyBatch
from ocf_datapipes.utils.utils import stack_np_examples_into_batch


@functional_datapipe("merge_numpy_examples_to_batch")
class MergeNumpyExamplesToBatchIterDataPipe(IterDataPipe):
    def __init__(self, source_datapipe: IterDataPipe, n_examples_per_batch: int):
        self.source_datapipe = source_datapipe
        self.n_examples_per_batch = n_examples_per_batch

    def __iter__(self) -> NumpyBatch:
        np_examples = []
        for np_batch in self.source_datapipe:
            np_examples.append(np_batch)
            if len(np_examples) == self.n_examples_per_batch:
                yield stack_np_examples_into_batch(np_examples)
                np_examples = []
