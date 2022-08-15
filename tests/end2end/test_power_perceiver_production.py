import torchdata.datapipes as dp

from ocf_datapipes.load import OpenGSP, OpenNWP, OpenPVFromNetCDF, OpenSatellite, OpenTopography
from ocf_datapipes.select import SelectOverlappingTimeSlice, SelectTimePeriods
from ocf_datapipes.transform.numpy import (
    AddSunPosition,
    AddTopographicData,
    AlignGSPto5Min,
    EncodeSpaceTime,
    SaveT0Time,
)
from ocf_datapipes.transform.xarray import (
    AddContiguousT0TimePeriods,
    ReduceNumPVSystems,
    SelectPVSystemsWithinRegion,
)


def test_power_perceiver_production():

    sat_datapipe = OpenSatellite()
    pv_datapipe = OpenPVFromNetCDF()
    gsp_datapipe = OpenGSP()
    nwp_datapipe = OpenNWP()
    topo_datapipe = OpenTopography()

    # Selecting overlapping time slices
    time_sat = AddContiguousT0TimePeriods(sat_datapipe)
    time_pv = AddContiguousT0TimePeriods(pv_datapipe)
    time_gsp = AddContiguousT0TimePeriods(gsp_datapipe)
    time_nwp = AddContiguousT0TimePeriods(nwp_datapipe)

    overlapping_time_datapipe = SelectOverlappingTimeSlice([time_pv, time_nwp, time_gsp, time_sat])
    # Now have overlapping time periods here

    pass
