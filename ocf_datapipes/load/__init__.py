"""Loading datapipes from the raw data"""
from ocf_datapipes.load.pv.live import OpenPVFromDBIterDataPipe as OpenPVFromDB
from ocf_datapipes.load.pv.pv import OpenPVFromNetCDFIterDataPipe as OpenPVFromNetCDF

from .configuration import OpenConfigurationIterDataPipe as OpenConfiguration
from .gsp import OpenGSPIterDataPipe as OpenGSP
from .nwp import OpenNWPIterDataPipe as OpenNWP
from .satellite import OpenSatelliteIterDataPipe as OpenSatellite

try:
    import rioxarray  # Rioxarray is sometimes a pain to install, so only load this if its installed

    from .topographic import OpenTopographyIterDataPipe as OpenTopography
except ImportError:
    print("Rioxarray is not installed, so not importing OpenTopography")
    pass
