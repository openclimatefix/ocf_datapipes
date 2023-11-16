from datetime import timedelta
from torch.utils.data.datapipes.iter import IterableWrapper
import numpy as np

from ocf_datapipes.transform.xarray import ApplyPVDropout


def test_apply_pv_dropout(passiv_datapipe):
    data = (
        next(iter(passiv_datapipe))
        .isel(pv_system_id=slice(0, 50))
        .isel(time_utc=slice(-10, None))
        .compute()
    )

    data = data.fillna(0)

    pv_datapipe = IterableWrapper([data for _ in range(3)])

    # ----------------
    # Apply no dropout
    pv_dropout_datapipe = ApplyPVDropout(
        pv_datapipe,
        system_dropout_fractions=[0],
        system_dropout_timedeltas=[timedelta(minutes=0)],
    )

    # No dropout should have been applied
    for pv_data in pv_dropout_datapipe:
        assert not np.isnan(pv_data.values).any()

    # --------------------------
    # Apply only system dropout
    pv_dropout_datapipe = ApplyPVDropout(
        pv_datapipe,
        system_dropout_fractions=[0.5],
        system_dropout_timedeltas=[timedelta(minutes=0)],
    )

    # Each system should have either all NaNs or no NaNs
    for pv_data in pv_dropout_datapipe:
        all_system_nan = pv_data.isnull().all(dim="time_utc")
        any_system_nan = pv_data.isnull().any(dim="time_utc")
        assert np.logical_or(all_system_nan.values, ~any_system_nan.values).all()

    # --------------------------
    # Apply only delay dropout
    pv_dropout_datapipe = ApplyPVDropout(
        pv_datapipe,
        system_dropout_fractions=[0.0],
        system_dropout_timedeltas=[timedelta(minutes=-5)],
    )

    # Each system should have 1 NaN
    for pv_data in pv_dropout_datapipe:
        assert (pv_data.isnull().sum(dim="time_utc") == 1).all()

    # --------------------------
    # Apply combo dropout
    pv_dropout_datapipe = ApplyPVDropout(
        pv_datapipe,
        system_dropout_fractions=[0.5],
        system_dropout_timedeltas=[timedelta(minutes=-5)],
    )

    # Each system should have either all NaNs or one NaNs
    for pv_data in pv_dropout_datapipe:
        all_system_nan = pv_data.isnull().all(dim="time_utc")
        one_system_nan = pv_data.isnull().sum(dim="time_utc") == 1
        assert np.logical_or(all_system_nan.values, one_system_nan.values).all()
