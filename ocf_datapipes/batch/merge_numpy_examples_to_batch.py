"""Merge individual examples into a batch"""

import logging
from typing import Sequence, Union

import numpy as np
from torch.utils.data import IterDataPipe, functional_datapipe

from ocf_datapipes.batch import BatchKey, NumpyBatch, NWPBatchKey, NWPNumpyBatch

logger = logging.getLogger(__name__)


def _key_is_constant(batch_key):
    is_constant = batch_key.name.endswith("t0_idx") or batch_key == NWPBatchKey.nwp_channel_names
    return is_constant


def stack_data_list(
    data_list: Sequence,
    batch_key: Union[BatchKey, NWPBatchKey],
):
    """How to combine data entries for each key

    See also: `extract_sample_from_batch()` for opposite
    """
    if _key_is_constant(batch_key):
        # These are always the same for all examples.
        return data_list[0]
    try:
        return np.stack(data_list)
    except Exception as e:
        logger.debug(f"Could not stack the following shapes together, ({batch_key})")
        shapes = [example.shape for example in data_list]
        logger.debug(shapes)
        logger.error(e)
        raise e


def extract_sample_from_batch(
    data,
    batch_key: Union[BatchKey, NWPBatchKey],
    index_num: int,
):
    """How to extract data entries for each key

    See also: `stack_data_list()` for opposite
    """
    if _key_is_constant(batch_key):
        # These are always the same for all examples.
        return data
    else:
        return data[index_num]


def stack_np_examples_into_batch(dict_list: Sequence[NumpyBatch]) -> NumpyBatch:
    """
    Stacks Numpy examples into a batch

    See also: `unstack_np_batch_into_examples()` for opposite

    Args:
        dict_list: A list of dict-like Numpy examples to stack

    Returns:
        The stacked NumpyBatch object
    """
    batch: NumpyBatch = {}

    batch_keys = list(dict_list[0].keys())

    for batch_key in batch_keys:
        # NWP is nested so treat separately
        if batch_key == BatchKey.nwp:
            nwp_batch: dict[str, NWPNumpyBatch] = {}

            # Unpack source keys
            nwp_sources = list(dict_list[0][BatchKey.nwp].keys())

            for nwp_source in nwp_sources:
                # Keys can be different for different NWPs
                nwp_batch_keys = list(dict_list[0][BatchKey.nwp][nwp_source].keys())

                nwp_source_batch: NWPNumpyBatch = {}
                for nwp_batch_key in nwp_batch_keys:
                    nwp_source_batch[nwp_batch_key] = stack_data_list(
                        [d[BatchKey.nwp][nwp_source][nwp_batch_key] for d in dict_list],
                        nwp_batch_key,
                    )

                nwp_batch[nwp_source] = nwp_source_batch

            batch[BatchKey.nwp] = nwp_batch

        else:
            batch[batch_key] = stack_data_list(
                [d[batch_key] for d in dict_list],
                batch_key,
            )

    return batch


def unstack_np_batch_into_examples(batch: NumpyBatch):
    """Splits a single batch into samples.

    Note:
    This can be really useful when using presaved batches, so you can split the samples, reshuffle,
    and recombine into batches. This means batches can be rebatched each epoch.

    See also: `stack_np_examples_into_batch()` for opposite
    """
    batch_keys = list(batch.keys())

    # Look at a non-constant key and find batch_size. Trickier if key is NWP
    non_constant_key = next(filter(lambda x: not _key_is_constant(x), batch_keys))
    if non_constant_key == BatchKey.nwp:
        # NWP is nested so treat separately
        nwp_source = next(iter(batch[BatchKey.nwp].keys()))
        non_constant_nwp_key = next(
            filter(
                lambda x: not _key_is_constant(x),
                list(batch[BatchKey.nwp][nwp_source].keys()),
            )
        )
        batch_size = batch[BatchKey.nwp][nwp_source][non_constant_nwp_key].shape[0]
    else:
        batch_size = batch[non_constant_key].shape[0]

    # Loop through and split the batch into samples
    samples = []
    for i in range(batch_size):
        sample: NumpyBatch = {}

        for key in batch_keys:
            # NWP is nested so treat separately
            if key == BatchKey.nwp:
                nwp_batch: dict[str, NWPNumpyBatch] = {}

                # Unpack keys
                nwp_sources = list(batch[BatchKey.nwp].keys())

                for nwp_source in nwp_sources:
                    nwp_source_batch: NWPNumpyBatch = {}

                    # Keys can be different for each NWP source
                    nwp_keys = list(batch[BatchKey.nwp][nwp_source].keys())

                    for nwp_key in nwp_keys:
                        nwp_source_batch[nwp_key] = extract_sample_from_batch(
                            batch[BatchKey.nwp][nwp_source][nwp_key],
                            batch_key=nwp_key,
                            index_num=i,
                        )

                    nwp_batch[nwp_source] = nwp_source_batch

                sample[BatchKey.nwp] = nwp_batch

            else:
                sample[key] = extract_sample_from_batch(
                    batch[key],
                    batch_key=key,
                    index_num=i,
                )

        samples += [sample]
    return samples


@functional_datapipe("merge_numpy_batch")
class MergeNumpyBatchIterDataPipe(IterDataPipe):
    """Merge list of individual examples into a batch"""

    def __init__(self, source_datapipe: IterDataPipe):
        """
        Merge list of individual examples into a batch

        Args:
            source_datapipe: Datapipe of yielding lists of numpybatch examples
        """
        self.source_datapipe = source_datapipe

    def __iter__(self) -> NumpyBatch:
        """Merge list of individual examples into a batch"""
        logger.debug("Merging numpy batch")
        for examples_list in self.source_datapipe:
            yield stack_np_examples_into_batch(examples_list)


# TODO: Is this needed anymore? Instead we can do either of:
#  - `dp.batch(batch_size).merge_numpy_batch()`
#  - `dp.batch(batch_size).map(stack_np_examples_into_batch)`
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
