"""Aligns the GSP data to 5 minutely as well"""
import numpy as np
import pandas as pd
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.utils.consts import BatchKey, NumpyBatch
from ocf_datapipes.utils.utils import datetime64_to_float, stack_np_examples_into_batch


@functional_datapipe("align_gsp_to_5_min")
class AlignGSPto5MinIterDataPipe(IterDataPipe):
    """Aligns the GSP data to 5 minutely as well"""

    def __init__(self, source_datapipe: IterDataPipe, batch_key_for_5_min_datetimes: BatchKey):
        """
        Aligns the GSP data to 5 minutely

        Args:
            source_datapipe: Datapipe of NumpyBatch
            batch_key_for_5_min_datetimes: What batch key to use for the 5 minute timestamps
        """
        self.source_datapipe = source_datapipe
        self.batch_key_for_5_min_datetimes = batch_key_for_5_min_datetimes

    def __iter__(self):
        for np_batch in self.source_datapipe:
            gsp_5_min_for_all_examples: list[NumpyBatch] = []
            n_examples = np_batch[BatchKey.gsp].shape[0]
            for example_i in range(n_examples):
                # Find the corresponding GSP 30 minute timestep
                # for each 5 minute satellite timestep.
                # We do this by taking the `ceil("30T")`
                # of each 5 minute satellite timestep.
                # Most of the code below is just converting to Pandas and back
                # so we can use `pd.DatetimeIndex.ceil` on each datetime:
                time_5_min = np_batch[self.batch_key_for_5_min_datetimes][example_i]
                time_5_min_dt_index = pd.to_datetime(time_5_min, unit="s")
                time_30_min_every_5_min_dt_index = time_5_min_dt_index.ceil("30T")
                time_30_min_every_5_min = datetime64_to_float(
                    time_30_min_every_5_min_dt_index.values
                )

                # Now, find the index into the original 30-minute GSP data for each 5-min timestep:
                gsp_30_min_time = np_batch[BatchKey.gsp_time_utc][example_i]
                idx_into_gsp = np.searchsorted(gsp_30_min_time, time_30_min_every_5_min)

                gsp_5_min_example: NumpyBatch = {}
                for batch_key in (BatchKey.gsp, BatchKey.gsp_time_utc):
                    new_batch_key_name = batch_key.name.replace("gsp", "gsp_5_min")
                    new_batch_key = BatchKey[new_batch_key_name]
                    gsp_5_min_example[new_batch_key] = np_batch[batch_key][example_i, idx_into_gsp]

                gsp_5_min_for_all_examples.append(gsp_5_min_example)

            # Stack the individual examples back into a batch of examples:
            new_np_batch = stack_np_examples_into_batch(gsp_5_min_for_all_examples)
            np_batch.update(new_np_batch)

            # Copy over the t0_idx scalar:
            batch_key_name_for_5_min_t0_idx = self.batch_key_for_5_min_datetimes.name.replace(
                "time_utc", "t0_idx"
            )
            batch_key_for_5_min_t0_idx = BatchKey[batch_key_name_for_5_min_t0_idx]
            np_batch[BatchKey.gsp_5_min_t0_idx] = np_batch[batch_key_for_5_min_t0_idx]

            yield np_batch
