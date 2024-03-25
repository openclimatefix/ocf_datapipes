"""Selects time slice from satellite, GSP data, or other xarray objects, and masks with dropout"""

from datetime import timedelta
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
from torch.utils.data import IterDataPipe, functional_datapipe


@functional_datapipe("draw_dropout_time")
class DrawDropoutTimeIterDataPipe(IterDataPipe):
    """Generates dropout times. The times are absolute values, not timedeltas."""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        dropout_timedeltas: List[timedelta],
        dropout_frac: Optional[float] = 0,
    ):
        """Generates dropout times. The times are absolute values, not timedeltas.

        Args:
            source_datapipe: Datapipe of t0 times
            dropout_timedeltas: List of timedeltas. We randonly select the delay for each time from
                this list. These should be negative timedeltas w.r.t time t0.
            dropout_frac: Fraction of samples subject to dropout
        """
        self.source_datapipe = source_datapipe
        self.dropout_timedeltas = dropout_timedeltas
        self.dropout_frac = dropout_frac
        if dropout_timedeltas is not None:
            assert len(dropout_timedeltas) >= 1, "Must include list of relative dropout timedeltas"
            assert all(
                [t < timedelta(minutes=0) for t in dropout_timedeltas]
            ), "dropout timedeltas must be negative"
        assert 0 <= dropout_frac <= 1

    def __iter__(self):
        for t0 in self.source_datapipe:
            if (self.dropout_timedeltas is None) or (np.random.uniform() >= self.dropout_frac):
                dropout_time = None
            else:
                t0_datetime_utc = pd.Timestamp(t0)
                dt = np.random.choice(self.dropout_timedeltas)
                dropout_time = t0_datetime_utc + dt

            yield dropout_time


@functional_datapipe("apply_dropout_time")
class ApplyDropoutTimeIterDataPipe(IterDataPipe):
    """Masks an xarray object to replace values that come after the dropout time with NaN."""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        dropout_time_datapipe: IterDataPipe,
    ):
        """Masks an xarray object to replace values that come after the dropout time with NaN.

        Args:
            source_datapipe: Datapipe of Xarray objects
            dropout_time_datapipe: Datapipe of dropout times
        """
        self.source_datapipe = source_datapipe
        self.dropout_time_datapipe = dropout_time_datapipe

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        for xr_data, dropout_time in self.source_datapipe.zip_ocf(self.dropout_time_datapipe):
            if dropout_time is None:
                yield xr_data
            else:
                # This replaces the times after the dropout with NaNs
                yield xr_data.where(xr_data.time_utc <= dropout_time)
