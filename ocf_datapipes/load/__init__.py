from .pv import OpenPVFromDBIterDataPipe as OpenPVFromDB
from .pv import OpenPVFromNetCDFIterDataPipe as OpenPVFromNetCDF
from .satellite import OpenSatelliteDataPipe as OpenSatellite
from .nwp import OpenNWPIterDataPipe as OpenNWP
from .gsp import OpenGSPIterDataPipe as OpenGSP

try:
    import rioxarray  # Rioxarray is sometimes a pain to install, so only load this if its installed
    from .topographic import OpenTopographyIterDataPipe as OpenTopography
except ImportError:
    print("Rioxarray is not installed, so not importing OpenTopography")
    pass
