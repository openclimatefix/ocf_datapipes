"""Loading datapipes from the raw data"""

from ocf_datapipes.load.gsp.database import OpenGSPFromDatabaseIterDataPipe as OpenGSPFromDatabase
from ocf_datapipes.load.gsp.gsp import OpenGSPIterDataPipe as OpenGSP
from ocf_datapipes.load.gsp.gsp_national import OpenGSPNationalIterDataPipe as OpenGSPNational
from ocf_datapipes.load.pv.database import OpenPVFromPVSitesDBIterDataPipe as OpenPVFromPVSitesDB
from ocf_datapipes.load.pv.pv import OpenPVFromNetCDFIterDataPipe as OpenPVFromNetCDF
from ocf_datapipes.load.sensor.awos import OpenAWOSFromNetCDFIterDataPipe as OpenAWOSFromNetCDF
from ocf_datapipes.load.sensor.meteomatics import (
    OpenMeteomaticsFromZarrIterDataPipe as OpenMeteomaticsFromZarr,
)
from ocf_datapipes.load.wind.wind import OpenWindFromNetCDFIterDataPipe as OpenWindFromNetCDF

from .configuration import OpenConfigurationIterDataPipe as OpenConfiguration
from .nwp.nwp import OpenNWPIterDataPipe as OpenNWP
from .satellite import OpenSatelliteIterDataPipe as OpenSatellite

try:
    import rioxarray  # Rioxarray is sometimes a pain to install, so only load this if its installed

    from .topographic import OpenTopographyIterDataPipe as OpenTopography
except ImportError:
    print("Rioxarray is not installed, so not importing OpenTopography")
    pass
