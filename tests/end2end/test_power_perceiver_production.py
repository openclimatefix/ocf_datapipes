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
    GetContiguousT0TimePeriods,
    ConvertSatelliteToInt8,
    Downsample,
    ReduceNumPVSystems,
)


def test_power_perceiver_production(sat_hrv_dp, passiv_dp, topo_dp, gsp_dp, nwp_dp):
    sat_hrv_dp = ConvertSatelliteToInt8(sat_hrv_dp)
    nwp_dp = Downsample(nwp_dp, y_coarsen=16, x_coarsen=16)

    pass
