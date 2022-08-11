from ocf_datapipes.load import OpenTopography, OpenNWP, OpenGSP, OpenSatellite, OpenPVFromNetCDF
from ocf_datapipes.transform.numpy import AlignGSPto5Min, AddSunPosition, AddTopographicData, SaveT0Time, EncodeSpaceTime
from ocf_datapipes.transform.xarray import SelectPVSystemsWithinRegion, ReduceNumPVSystems

def test_power_perceiver_production():
    pass
