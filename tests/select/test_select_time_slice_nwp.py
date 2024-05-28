from datetime import timedelta
from torch.utils.data.datapipes.iter import IterableWrapper
import pandas as pd

from ocf_datapipes.select import SelectTimeSliceNWP


def test_select_time_slice_nwp(nwp_datapipe):
    ds_nwp = next(iter(nwp_datapipe))

    t0 = pd.Timestamp(ds_nwp.init_time_utc.values[3])
    times = [t0+timedelta(minutes=m) for m in [-60, 0, 30, 60, 120]]
    
    # Dropout with one 5 minute delay
    dropout_datapipe = SelectTimeSliceNWP(
        nwp_datapipe,
        IterableWrapper(times),
        sample_period_duration=timedelta(minutes=60),
        history_duration=timedelta(minutes=60),
        forecast_duration=timedelta(minutes=60),
        dropout_timedeltas=[timedelta(minutes=-5)],
        dropout_frac=1,
    )

    # The init-times should be the last 3-hour mutliple starting 5-minutes before each time t
    for t, ds in zip(times, dropout_datapipe):
        assert "target_time_utc" in ds.coords
        assert (
            ds.init_time_utc.values == (t - timedelta(minutes=5)).floor(timedelta(hours=3))
        ).all()
        
    times = [t0+timedelta(minutes=m) for m in [0, 30, 120]]

    # Dropout with one hour delay
    dropout_datapipe = SelectTimeSliceNWP(
        nwp_datapipe,
        IterableWrapper(times),
        sample_period_duration=timedelta(minutes=60),
        history_duration=timedelta(minutes=60),
        forecast_duration=timedelta(minutes=0),
        dropout_timedeltas=[timedelta(minutes=-60)],
        dropout_frac=1,
    )

    # The init times should be the last 3-hour mutliple starting an hour before each time t
    for t, ds in zip(times, dropout_datapipe):
        assert "target_time_utc" in ds.coords
        assert (
            ds.init_time_utc.values == (t - timedelta(minutes=60)).floor(timedelta(hours=3))
        ).all()


def test_select_time_slice_nwp_diff(nwp_datapipe):
    ds_nwp = next(iter(nwp_datapipe))
    
    t0 = pd.Timestamp(ds_nwp.init_time_utc.values[3])
    times = [t0+timedelta(minutes=m) for m in [0, 30, 120]]

    # No diffing
    dropout_datapipe = SelectTimeSliceNWP(
        nwp_datapipe,
        IterableWrapper(times),
        sample_period_duration=timedelta(minutes=60),
        history_duration=timedelta(minutes=60),
        forecast_duration=timedelta(minutes=0),
        dropout_timedeltas=[timedelta(minutes=-60)],
        dropout_frac=1,
    )        
    
    # With diffing
    dropout_datapipe_diffed = SelectTimeSliceNWP(
        nwp_datapipe,
        IterableWrapper(times),
        sample_period_duration=timedelta(minutes=60),
        history_duration=timedelta(minutes=60),
        forecast_duration=timedelta(minutes=0),
        dropout_timedeltas=[timedelta(minutes=-60)],
        dropout_frac=1,
        accum_channels=[ds_nwp.channel.values[0]]
    )

    # The init times should be the last 3-hour mutliple starting an hour before each time t
    for ds, ds_diffed in zip(dropout_datapipe, dropout_datapipe_diffed):
        
        # Diff the un-diffed data and select part
        ds1 = (
            ds.sel(channel=[ds_nwp.channel.values[0]])
            .diff(dim="target_time_utc", label="lower")
            .isel(target_time_utc=slice(0, 1))
            .compute()
        )
        
        # Select same part of diffed data
        ds2 = (
            ds_diffed.sel(channel=[ds_nwp.channel.values[0]])
            .isel(target_time_utc=slice(0, 1))
            .compute()
        )
        
        # Check they are equal
        assert ds1.equals(ds2)
