import logging

import numpy as np
import pandas as pd
import xarray as xr

from ocf_datapipes.select import TrimDatesWithInsufficentData, RemoveBadSystems
from ocf_datapipes.select import SelectSysWithoutOutputWholeday as SysWithoutPV


# TODO Still needed to work on this tests a bit
def test_sys_with_lessthan_oneday_pv(passiv_datapipe):
    data = TrimDatesWithInsufficentData(passiv_datapipe)
    data = next(iter(data))
    count = len(data.coords["time_utc"].values)
    assert count % 288. == 0.

        # Removes any PV systems with less than 1 day of data.

        # This is done, by counting all the nan values and check the
        # count is greater than 289 (number of 5 minute intervals in a day)

# def test_sys_without_pv(passiv_datapipe):
#     no_pv_wholeday = SysWithoutPV(passiv_datapipe)
#     data = next(iter(no_pv_wholeday))
#     key1 = sorted(data.keys())
#     key2 = sorted(list({k2 for v in data.values() for k2 in v}))
#     for x in key1:
#         for y in key2:
#             assert data[x][y] == "Active" or "Inactive"


# def test_sys_without_passivpv(passiv_datapipe):
#     no_pv_wholeday = RemoveBadSystems(passiv_datapipe)
#     data = next(iter(no_pv_wholeday))
#     assert len(list(data)) != 0
