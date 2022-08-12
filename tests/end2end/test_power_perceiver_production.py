import torchdata.datapipes as dp

from ocf_datapipes.load import OpenGSP, OpenNWP, OpenPVFromNetCDF, OpenSatellite, OpenTopography
from ocf_datapipes.transform.numpy import (
    AddSunPosition,
    AddTopographicData,
    AlignGSPto5Min,
    EncodeSpaceTime,
    SaveT0Time,
)
from ocf_datapipes.transform.xarray import ReduceNumPVSystems, SelectPVSystemsWithinRegion
from ocf_datapipes.select import SelectTimePeriods, SelectOverlappingTimeSlice


def test_power_perceiver_production():

    sat_datapipe = OpenSatellite()
    pv_datapipe = OpenPVFromNetCDF()
    gsp_datapipe = OpenGSP()
    nwp_datapipe = OpenNWP()
    topo_datapipe = OpenTopography()

    pass
