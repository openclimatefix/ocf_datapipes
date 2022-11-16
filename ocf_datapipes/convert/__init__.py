"""Various conversion datapipes"""
from ocf_datapipes.convert.numpy.batch.gsp import (
    ConvertGSPToNumpyBatchIterDataPipe as ConvertGSPToNumpyBatch,
)
from ocf_datapipes.convert.numpy.batch.pv import (
    ConvertPVToNumpyBatchIterDataPipe as ConvertPVToNumpyBatch,
)

from .coordinates import ConvertGeostationaryToLatLonIterDataPipe as ConvertGeostationaryToLatLon
from .coordinates import ConvertLatLonToOSGBIterDataPipe as ConvertLatLonToOSGB
from .coordinates import ConvertOSGBToLatLonIterDataPipe as ConvertOSGBToLatLon
from .numpy.gsp import ConvertGSPToNumpyIterDataPipe as ConvertGSPToNumpy
from .numpy.pv import ConvertPVToNumpyIterDataPipe as ConvertPVToNumpy
from .nwp import ConvertNWPToNumpyBatchIterDataPipe as ConvertNWPToNumpyBatch
from .satellite import ConvertSatelliteToNumpyBatchIterDataPipe as ConvertSatelliteToNumpyBatch
