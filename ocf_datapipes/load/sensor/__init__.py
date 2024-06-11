"""Loading sensor data (i.e. weather station, rainfall radar, etc.)"""

from .awos import OpenAWOSFromNetCDFIterDataPipe as OpenAWOSFromNetCDF
from .meteomatics import OpenMeteomaticsFromZarrIterDataPipe as OpenMeteomaticsFromZarr
