"""Various conversion datapipes"""
from ocf_datapipes.convert.numpy_batch.gsp import (
    ConvertGSPToNumpyBatchIterDataPipe as ConvertGSPToNumpyBatch,
)
from ocf_datapipes.convert.numpy_batch.nwp import (
    ConvertNWPToNumpyBatchIterDataPipe as ConvertNWPToNumpyBatch,
)
from ocf_datapipes.convert.numpy_batch.pv import (
    ConvertPVToNumpyBatchIterDataPipe as ConvertPVToNumpyBatch,
)
from ocf_datapipes.convert.numpy_batch.satellite import (
    ConvertSatelliteToNumpyBatchIterDataPipe as ConvertSatelliteToNumpyBatch,
)
from ocf_datapipes.convert.numpy_batch.sensor import (
    ConvertSensorToNumpyBatchIterDataPipe as ConvertSensorToNumpyBatch,
)

from .coordinates import ConvertGeostationaryToLonLatIterDataPipe as ConvertGeostationaryToLonLat
from .coordinates import ConvertLonLatToOSGBIterDataPipe as ConvertLonLatToOSGB
from .coordinates import ConvertOSGBToLonLatIterDataPipe as ConvertOSGBToLonLat
from .numpy.gsp import ConvertGSPToNumpyIterDataPipe as ConvertGSPToNumpy
from .numpy.pv import ConvertPVToNumpyIterDataPipe as ConvertPVToNumpy
