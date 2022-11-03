"""Various conversion datapipes"""
from ocf_datapipes.convert.numpy.batch.gsp import (
    ConvertGSPToNumpyBatchIterDataPipe as ConvertGSPToNumpyBatch,
)

from .coordinates import ConvertGeostationaryToLatLonIterDataPipe as ConvertGeostationaryToLatLon
from .coordinates import ConvertLatLonToOSGBIterDataPipe as ConvertLatLonToOSGB
from .coordinates import ConvertOSGBToLatLonIterDataPipe as ConvertOSGBToLatLon
from .numpy.gsp import ConvertGSPToNumpyIterDataPipe as ConvertGSPToNumpy
from .nwp import ConvertNWPToNumpyBatchIterDataPipe as ConvertNWPToNumpyBatch
from .pv import ConvertPVToNumpyBatchIterDataPipe as ConvertPVToNumpyBatch
from .satellite import ConvertSatelliteToNumpyBatchIterDataPipe as ConvertSatelliteToNumpyBatch
