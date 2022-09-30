"""Merge individual examples into a batch"""
import logging

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.utils.consts import NumpyBatch
from ocf_datapipes.utils.utils import stack_np_examples_into_batch

logger = logging.getLogger(__name__)


@functional_datapipe("merge_numpy_examples_to_batch")
class MergeNumpyExamplesToBatchIterDataPipe(IterDataPipe):
    """Merge individual examples into a batch"""

    def __init__(self, source_datapipe: IterDataPipe, n_examples_per_batch: int):
        """
        Merge individual examples into a batch

        Args:
            source_datapipe: Datapipe of NumpyBatch data
            n_examples_per_batch: Number of examples per batch
        """
        self.source_datapipe = source_datapipe
        self.n_examples_per_batch = n_examples_per_batch

    def __iter__(self) -> NumpyBatch:
        """Merge individual examples into a batch"""
        np_examples = []
        logger.debug("Merging numpy batch")
        for np_batch in self.source_datapipe:
            np_examples.append(np_batch)
            if len(np_examples) == self.n_examples_per_batch:
                yield stack_np_examples_into_batch(np_examples)
                np_examples = []
