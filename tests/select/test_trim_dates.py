import numpy as np
import pandas as pd
import xarray as xr

from ocf_datapipes.select import TrimDatesWithInsufficentData


def test_trim_lessthan_oneday(passiv_datapipe):
    data = TrimDatesWithInsufficentData(passiv_datapipe, minimum_number_data_points=288)
    data = next(iter(data))
    count = len(data.coords["time_utc"].values)
    assert count % 288 == 0


def test_with_pvoutput_datapipe(pvoutput_datapipe):
    after_trim_date = TrimDatesWithInsufficentData(
        pvoutput_datapipe, minimum_number_data_points=288
    )

    before_data = next(iter(pvoutput_datapipe))
    after_data = next(iter(after_trim_date))

    assert len(before_data.coords["time_utc"].values) == len(after_data.coords["time_utc"].values)


def test_constructed_xarray():
    def construct_multi_daterange(freq: str, periods: np.int32, start: str = "2022-01-01"):
        time_range = pd.date_range(start=start, freq=freq, periods=periods)
        pv_system_id = [1, 2, 3]
        ALL_COORDS = {"time_utc": time_range, "pv_system_id": pv_system_id}

        data = np.zeros((len(time_range), len(pv_system_id)))
        data[:, 2] = np.nan

        data_array = xr.DataArray(data, coords=ALL_COORDS,)
        return data_array

    # Different data array with different time ranges
    # Data_array1 extends to a second day with five minute intervals
    # Data_array2 does not have full day (30min) intervals
    # Data_array3 extends more than ~8days
    data_array1 = construct_multi_daterange(freq="5T", periods=350)
    data_array2 = construct_multi_daterange(freq="30T", periods=25)
    data_array3 = construct_multi_daterange(freq="5T", periods=2500)
    trim_date1_5min_interval = TrimDatesWithInsufficentData(
        [data_array1], minimum_number_data_points=288
    )
    trim_date2_30min_interval = TrimDatesWithInsufficentData(
        [data_array2], minimum_number_data_points=48
    )
    trim_date3_5min_interval = TrimDatesWithInsufficentData(
        [data_array3], minimum_number_data_points=288
    )

    data1 = next(iter(trim_date1_5min_interval))
    data2 = next(iter(trim_date2_30min_interval))
    data3 = next(iter(trim_date3_5min_interval))

    # Function slices data_array1 dates to a single day
    # Function does not slice any dates in data_array2,
    # as it has dates less than a day
    # Function slices data_array3 dates to the 12th hour and,
    # remaining dates are a multiple of 5-minute intervals in a day (288)
    assert len(data1.coords["time_utc"].values) == 288
    assert len(data2.coords["time_utc"].values) == 25
    assert len(data3.coords["time_utc"].values) % 288 == 0.0
