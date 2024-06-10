"""Convert NWP data to the target time with dropout"""

import logging
from datetime import timedelta
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
from torch.utils.data import IterDataPipe, functional_datapipe

logger = logging.getLogger(__name__)


@functional_datapipe("select_time_slice_nwp")
class SelectTimeSliceNWPIterDataPipe(IterDataPipe):
    """Convert NWP Xarray dataset to use target time as indexer"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        t0_datapipe: IterDataPipe,
        sample_period_duration: timedelta,
        history_duration: timedelta,
        forecast_duration: timedelta,
        dropout_timedeltas: Optional[List[timedelta]] = None,
        dropout_frac: Optional[float] = 0,
        accum_channels: Optional[List[str]] = [],
        channel_dim_name: str = "channel",
    ):
        """Convert NWP Xarray dataset to use target time as indexer

        Args:
            source_datapipe: Datapipe emitting an Xarray Dataset with step and init_time_utc
                indexers.
            t0_datapipe: Datapipe emitting t0 times for indexing off of choosing the closest
                previous init_time_utc.
            sample_period_duration: How long the sampling period is.
            history_duration: How long the history time should cover.
            forecast_duration: How long the forecast time should cover.
            dropout_timedeltas: List of timedeltas. We randonly select the delay for each NWP
                forecast from this list. These should be negative timedeltas w.r.t time t0.
            dropout_frac: Fraction of samples subject to dropout
            accum_channels: Some variables which are stored as accumulations. This allows us to take
                the diff of these channels.
            channel_dim_name: Dimension name for channels
        """
        self.source_datapipe = source_datapipe
        self.t0_datapipe = t0_datapipe
        self.sample_period_duration = sample_period_duration
        self.history_duration = history_duration
        self.forecast_duration = forecast_duration
        self.dropout_timedeltas = dropout_timedeltas
        self.dropout_frac = dropout_frac
        self.accum_channels = accum_channels
        self.channel_dim_name = channel_dim_name

        if dropout_timedeltas is not None:
            assert all(
                [t < timedelta(minutes=0) for t in dropout_timedeltas]
            ), "dropout timedeltas must be negative"
            assert len(dropout_timedeltas) >= 1
        assert 0 <= dropout_frac <= 1
        self._consider_dropout = (dropout_timedeltas is not None) and dropout_frac > 0

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        """Iterate through both datapipes and convert Xarray dataset"""

        for t0, xr_data in self.t0_datapipe.zip(self.source_datapipe):
            # The accumatation and non-accumulation channels
            accum_channels = np.intersect1d(
                xr_data[self.channel_dim_name].values, self.accum_channels
            )
            non_accum_channels = np.setdiff1d(
                xr_data[self.channel_dim_name].values, self.accum_channels
            )

            t0 = pd.Timestamp(t0)
            start_dt = (t0 - self.history_duration).ceil(self.sample_period_duration)
            end_dt = (t0 + self.forecast_duration).ceil(self.sample_period_duration)

            target_times = pd.date_range(start_dt, end_dt, freq=self.sample_period_duration)

            # Maybe apply NWP dropout
            if self._consider_dropout and (np.random.uniform() < self.dropout_frac):
                dt = np.random.choice(self.dropout_timedeltas)
                t0_available = t0 + dt
            else:
                t0_available = t0

            # Forecasts made up to and including t0
            available_init_times = xr_data.init_time_utc.sel(
                init_time_utc=slice(None, t0_available)
            )

            # Find the most recent available init times for all target times
            selected_init_times = available_init_times.sel(
                init_time_utc=target_times,
                method="ffill",  # forward fill from init times to target times
            ).values

            # Find the required steps for all target times
            steps = target_times - selected_init_times

            # We want one timestep for each target_time_hourly (obviously!) If we simply do
            # nwp.sel(init_time=init_times, step=steps) then we'll get the *product* of
            # init_times and steps, which is not what # we want! Instead, we use xarray's
            # vectorized-indexing mode by using a DataArray indexer.  See the last example here:
            # https://docs.xarray.dev/en/latest/user-guide/indexing.html#more-advanced-indexing
            coords = {"target_time_utc": target_times}
            init_time_indexer = xr.DataArray(selected_init_times, coords=coords)
            step_indexer = xr.DataArray(steps, coords=coords)

            if len(accum_channels) == 0:
                xr_sel = xr_data.sel(step=step_indexer, init_time_utc=init_time_indexer)

            else:
                # First minimise the size of the dataset we are diffing
                # - find the init times we are slicing from
                unique_init_times = np.unique(selected_init_times)
                # - find the min and max steps we slice over. Max is extended due to diff
                min_step = min(steps)
                max_step = max(steps) + (xr_data.step[1] - xr_data.step[0])

                xr_min = xr_data.sel(
                    {
                        "init_time_utc": unique_init_times,
                        "step": slice(min_step, max_step),
                    }
                )

                # Slice out the data which does not need to be diffed
                xr_non_accum = xr_min.sel({self.channel_dim_name: non_accum_channels})
                xr_sel_non_accum = xr_non_accum.sel(
                    step=step_indexer, init_time_utc=init_time_indexer
                )

                # Slice out the channels which need to be diffed
                xr_accum = xr_min.sel({self.channel_dim_name: accum_channels})

                # Take the diff and slice requested data
                xr_accum = xr_accum.diff(dim="step", label="lower")
                xr_sel_accum = xr_accum.sel(step=step_indexer, init_time_utc=init_time_indexer)

                # Join diffed and non-diffed variables
                xr_sel = xr.concat([xr_sel_non_accum, xr_sel_accum], dim=self.channel_dim_name)

                # Reorder the variable back to the original order
                xr_sel = xr_sel.sel({self.channel_dim_name: xr_data[self.channel_dim_name].values})

                # Rename the diffed channels
                xr_sel[self.channel_dim_name] = [
                    f"diff_{v}" if v in accum_channels else v
                    for v in xr_sel[self.channel_dim_name].values
                ]

            yield xr_sel
