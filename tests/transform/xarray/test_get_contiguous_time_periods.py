from ocf_datapipes.transform.xarray import GetContiguousT0TimePeriods
from ocf_datapipes.select import LocationPicker, DropGSP
from datetime import timedelta

def test_get_contiguous_time_periods(nwp_datapipe):
    nwp_datapipe = GetContiguousT0TimePeriods(nwp_datapipe,
        sample_period_duration=timedelta(hours=3),
        history_duration=timedelta(minutes=60),
        forecast_duration=timedelta(minutes=180),
        time_dim="init_time_utc",
    )

    batch = next(iter(nwp_datapipe))
    print(batch)
