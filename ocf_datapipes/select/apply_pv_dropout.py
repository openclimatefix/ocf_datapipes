"""Convert NWP data to the target time with dropout"""

import logging
from datetime import timedelta
from typing import List, Optional, Union

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
        min_frac: float,
        max_frac: float,
        system_dropout_timedeltas: Optional[List[timedelta]],
    ):
        """Apply PV system dropout to mimic production

        Systems have independent delay times. Systems may also completely drop out.

        For each yielded sample, a dropout fraction will be randomly chosen from the range
        [min_frac, max_frac]. This random fraction will be used as the chance of dropout for each
        PV system in the sample. Using a list instead of a single value allows us to avoid
        overfitting to the fraction of dropped out systems.

        Args:
            source_datapipe: Datapipe emitting an Xarray Dataset with time_utc indexer.
            min_frac: The minimum chance for each system to be dropped out in a sample
            max_frac: The maximum chance for each system to be dropped out in a sample
            system_dropout_timedeltas: List of timedeltas. We randomly select the delay for each PV
                system from this list. These should be negative timedeltas w.r.t the last time_utc
                value of the xarray data.
        """
        self.source_datapipe = source_datapipe
        self.min_frac = min_frac
        self.max_frac = max_frac
        self.system_dropout_timedeltas = system_dropout_timedeltas

        if system_dropout_timedeltas is not None:
            assert (
                len(system_dropout_timedeltas) >= 1
            ), "Must include list of relative dropout timedeltas"

            assert all(
                [t <= timedelta(minutes=0) for t in system_dropout_timedeltas]
            ), f"dropout timedeltas must be negative: {system_dropout_timedeltas}"

        assert all(
            [0 <= f <= 1 for f in [min_frac, max_frac]]
        ), "The min and max dropout fractions must be in open range (0, 1)"

        assert min_frac <= max_frac, "Min dropout fraction <= maximum dropout fraction"

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        """Iterate through Xarray dataset using dropout"""

        for xr_data in self.source_datapipe:
            # Assign these values for convenience
            t0 = pd.Timestamp(xr_data.time_utc.values[-1])
            n_systems = len(xr_data.pv_system_id)

            if not (self.min_frac == self.max_frac == 0):
                # Apply PV system dropout - individual systems are dropped out

                # Don't want fraction of dropped out system to be the same in each sample
                # This might lead to overfitting. Instead vary it
                dropout_p = np.random.uniform(low=self.min_frac, high=self.max_frac)

                system_mask = xr.zeros_like(xr_data.pv_system_id, dtype=bool)
                system_mask.values[:] = np.random.uniform(size=n_systems) >= dropout_p

                xr_data = xr_data.where(system_mask)

            if self.system_dropout_timedeltas is not None:
                # Apply independent delay to each PV system
                delay_mask = xr.zeros_like(xr_data, dtype=bool)

                last_available_times = xr.zeros_like(
                    xr_data.pv_system_id, dtype=xr_data.time_utc.dtype
                )
                last_available_times.values[:] = t0 + np.random.choice(
                    self.system_dropout_timedeltas, size=n_systems
                )

                delay_mask = xr_data.time_utc <= last_available_times

                # Apply masking
                xr_data = xr_data.where(delay_mask)

            yield xr_data
