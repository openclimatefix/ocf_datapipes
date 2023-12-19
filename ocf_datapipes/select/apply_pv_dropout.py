"""Convert NWP data to the target time with dropout"""
import logging
from datetime import timedelta
from typing import List, Union

import numpy as np
import pandas as pd
import xarray as xr
from torch.utils.data import IterDataPipe, functional_datapipe

logger = logging.getLogger(__name__)


@functional_datapipe("apply_pv_dropout")
class ApplyPVDropoutIterDataPipe(IterDataPipe):
    """Apply PV system dropout to mimic production

    Systems have independent delay times. Systems may also completely drop out.

    """

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        system_dropout_fractions: List[float],
        system_dropout_timedeltas: List[timedelta],
    ):
        """Apply PV system dropout to mimic production

        Systems have independent delay times. Systems may also completely drop out.

        Args:
            source_datapipe: Datapipe emitting an Xarray Dataset with time_utc indexer.
            system_dropout_fractions: List of possible system dropout fractions to apply to each
                sample. For each yielded sample, one of these fractions will be chosen and used to
                dropout each PV system. Using a list instead of a single value allows us to avoid
                overfitting to the fraction of dropped out systems.
            system_dropout_timedeltas: List of timedeltas. We randomly select the delay for each PV
                system from this list. These should be negative timedeltas w.r.t the last time_utc
                value of the xarray data.
        """
        self.source_datapipe = source_datapipe
        self.system_dropout_fractions = system_dropout_fractions
        self.system_dropout_timedeltas = system_dropout_timedeltas

        assert (
            len(system_dropout_timedeltas) >= 1
        ), "Must include list of relative dropout timedeltas"

        assert all(
            [t <= timedelta(minutes=0) for t in system_dropout_timedeltas]
        ), f"dropout timedeltas must be negative: {system_dropout_timedeltas}"

        assert all(
            [0 <= f <= 1 for f in system_dropout_fractions]
        ), "dropout fractions must be in open range (0, 1)"

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        """Iterate through Xarray dataset using dropout"""

        for xr_data in self.source_datapipe:
            # Assign these values for convenience
            t0 = pd.Timestamp(xr_data.time_utc.values[-1])
            n_systems = len(xr_data.pv_system_id)

            # Apply PV system dropout - individual systems are dropped out

            # Don't want fraction of dropped out system to be the same in each sample
            # This might lead to overfitting. Instead vary it
            dropout_p = np.random.choice(self.system_dropout_fractions)

            system_mask = xr.zeros_like(xr_data.pv_system_id, dtype=bool)
            system_mask.values[:] = np.random.uniform(size=n_systems) >= dropout_p

            # Apply independent delay to each PV system
            delay_mask = xr.zeros_like(xr_data, dtype=bool)

            last_available_times = xr.zeros_like(xr_data.pv_system_id, dtype=xr_data.time_utc.dtype)
            last_available_times.values[:] = t0 + np.random.choice(
                self.system_dropout_timedeltas, size=n_systems
            )

            delay_mask = xr_data.time_utc <= last_available_times

            # Apply masking
            xr_data = xr_data.where(system_mask).where(delay_mask)

            yield xr_data
