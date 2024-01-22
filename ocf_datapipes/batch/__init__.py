"""Datapipes for batching together data"""
from .batches import BatchKey, NumpyBatch, NWPBatchKey, NWPNumpyBatch, XarrayBatch
from .merge_numpy_examples_to_batch import (
    MergeNumpyBatchIterDataPipe as MergeNumpyBatch,
)
from .merge_numpy_examples_to_batch import (
    MergeNumpyExamplesToBatchIterDataPipe as MergeNumpyExamplesToBatch,
)
from .merge_numpy_examples_to_batch import (
    stack_np_examples_into_batch,
    unstack_np_batch_into_examples,
)
from .merge_numpy_modalities import MergeNumpyModalitiesIterDataPipe as MergeNumpyModalities
from .merge_numpy_modalities import MergeNWPNumpyModalitiesIterDataPipe as MergeNWPNumpyModalities
