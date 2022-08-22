import numpy as np
import pandas as pd
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.utils.consts import NumpyBatch


@functional_datapipe("extend_timesteps_to_future")
class ExtendTimestepsToFutureIterDataPipe(IterDataPipe):
    def __init__(
        self, source_dp: IterDataPipe, modality, forecast_duration, sample_period_duration
    ):
        """This assumes that the current time_utc array only covers history + now, so just extends it further into the future"""
        self.source_dp = source_dp
        self.modality = modality
        self.forecast_duration = forecast_duration
        self.sample_period_duration = sample_period_duration

    def __iter__(self) -> NumpyBatch:
        for np_batch in self.source_dp:

            yield np_batch
