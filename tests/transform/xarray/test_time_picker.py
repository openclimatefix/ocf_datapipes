from datetime import timedelta

from ocf_datapipes.select import SelectOverlappingTimeSlice, SelectTimePeriods, SelectTimeSlice
from ocf_datapipes.transform.xarray import GetContiguousT0TimePeriods


def test_time_picker(sat_hrv_dp, passiv_dp, gsp_dp):
    sat_time_dp = GetContiguousT0TimePeriods(sat_hrv_dp)
    pv_time_dp = GetContiguousT0TimePeriods(passiv_dp)
    gsp_time_dp = GetContiguousT0TimePeriods(gsp_dp)
