"""Various conversion datapipes"""
from .numpy_batch.gsp import (
    ConvertGSPToNumpyBatchIterDataPipe as ConvertGSPToNumpyBatch,
)
from .numpy_batch.nwp import (
    ConvertNWPToNumpyBatchIterDataPipe as ConvertNWPToNumpyBatch,
)
from .numpy_batch.pv import (
    ConvertPVToNumpyBatchIterDataPipe as ConvertPVToNumpyBatch,
)
from .numpy_batch.satellite import (
    ConvertSatelliteToNumpyBatchIterDataPipe as ConvertSatelliteToNumpyBatch,
)
from .numpy_batch.sensor import (
    ConvertSensorToNumpyBatchIterDataPipe as ConvertSensorToNumpyBatch,
)
from .convert_to_numpy_stack import StackXarrayIterDataPipe as StackXarray

from .coordinates import ConvertGeostationaryToLonLatIterDataPipe as ConvertGeostationaryToLonLat
from .coordinates import ConvertLonLatToOSGBIterDataPipe as ConvertLonLatToOSGB
from .coordinates import ConvertOSGBToLonLatIterDataPipe as ConvertOSGBToLonLat
from .numpy.gsp import ConvertGSPToNumpyIterDataPipe as ConvertGSPToNumpy
from .numpy.pv import ConvertPVToNumpyIterDataPipe as ConvertPVToNumpy
