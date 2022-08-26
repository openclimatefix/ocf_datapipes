from typing import Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@functional_datapipe("pv_power_rolling_window")
class PVPowerRollingWindowIterDataPipe(IterDataPipe):
    def __init__(
        self,
        source_dp: IterDataPipe,
        window: Union[int, pd.tseries.offsets.DateOffset, pd.core.indexers.objects.BaseIndexer] = 3,
        min_periods: Optional[int] = 2,
        center: bool = True,
        win_type: Optional[str] = None,
        expect_dataset: bool = True,
    ):
        self.source_dp = source_dp
        self.window = window
        self.min_periods = min_periods
        self.center = center
        self.win_type = win_type
        self.expect_dataset = expect_dataset

    def __iter__(self):
        for xr_data in self.source_dp:
            if self.expect_dataset:
                data_to_resample = xr_data["power_w"]
            else:
                data_to_resample = xr_data

            resampled = data_to_resample.rolling(
                dim={"time_utc": self.window},
                min_periods=self.min_periods,
                center=self.center,
            ).mean()

            if self.expect_dataset:
                xr_data["power_w"] = resampled
                resampled = xr_data

            # Resampling removes the attributes, so put them back:
            for attr_name in ("t0_idx", "sample_period_duration"):
                resampled.attrs[attr_name] = xr_data.attrs[attr_name]

            yield resampled


def set_new_sample_period_and_t0_idx_attrs(xr_data, new_sample_period) -> xr.DataArray:
    orig_sample_period = xr_data.attrs["sample_period_duration"]
    orig_t0_idx = xr_data.attrs["t0_idx"]
    new_sample_period = pd.Timedelta(new_sample_period)
    assert new_sample_period >= orig_sample_period
    new_t0_idx = orig_t0_idx / (new_sample_period / orig_sample_period)
    np.testing.assert_almost_equal(
        int(new_t0_idx),
        new_t0_idx,
        err_msg=(
            "The original t0_idx must be exactly divisible by"
            " (new_sample_period / orig_sample_period)"
        ),
    )
    xr_data.attrs["sample_period_duration"] = new_sample_period
    xr_data.attrs["t0_idx"] = int(new_t0_idx)
    return xr_data
