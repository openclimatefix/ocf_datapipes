from ocf_datapipes.load import OpenTopography, OpenNWP, OpenGSP, OpenSatellite, OpenPVFromNetCDF
from ocf_datapipes.transform.numpy import AlignGSPto5Min, AddSunPosition, AddTopographicData, SaveT0Time, EncodeSpaceTime
from ocf_datapipes.transform.xarray import SelectPVSystemsWithinRegion, ReduceNumPVSystems
import torchdata.datapipes as dp

def test_power_perceiver_production():

    sat_datapipe = OpenSatellite()
    pv_datapipe = OpenPVFromNetCDF()
    gsp_datapipe = OpenGSP()
    nwp_datapipe = OpenNWP()
    topo_datapipe = OpenTopography()


    pass
